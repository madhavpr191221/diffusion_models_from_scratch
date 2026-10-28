"""Local API for the pre-training forward-noise viewer."""

from __future__ import annotations

import io
import json
import os
import re
import threading
import time
import tomllib
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from typing import Literal
from uuid import uuid4

import numpy as np
import torch
from fastapi import BackgroundTasks, FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, Field
from PIL import Image
from openai import OpenAI
from dotenv import load_dotenv

from ddpm_eurosat.data import UnconditionalEuroSATSplit, image_to_tensor, load_eurosat_rgb
from ddpm_eurosat.diffusion import DiffusionSchedule, q_sample, q_sample_markov
from ddpm_eurosat.generation_evaluation import (
    GENERATION_CHUNK_SIZE,
    JUDGEMENT_BATCH_SIZE,
    REFERENCE_COUNT,
    build_judgement_report,
    generate_cohort,
    judge_batch,
)
from ddpm_eurosat.sampling import LIVE_FRAME_STRIDE, choose_device, display_grid, generate_from_checkpoint
from ddpm_eurosat.score_sde_sampling import generate_from_checkpoint as generate_from_vp_sde_checkpoint

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env", override=False)
CONFIG = tomllib.loads((PROJECT_ROOT / "configs" / "base.toml").read_text(encoding="utf-8"))
DIFFUSION_CONFIG = CONFIG["diffusion"]
DATASET_CONFIG = CONFIG["dataset"]
SCHEDULE = DiffusionSchedule(
    timesteps=int(DIFFUSION_CONFIG["timesteps"]),
    beta_start=float(DIFFUSION_CONFIG["beta_start"]),
    beta_end=float(DIFFUSION_CONFIG["beta_end"]),
)

app = FastAPI(title="Unconditional EuroSAT DDPM viewer")
RUNS_ROOT = PROJECT_ROOT / "runs"


def get_run_dir(run_id: str) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,100}", run_id):
        raise HTTPException(status_code=404, detail="Run not found")
    run_dir = RUNS_ROOT / run_id
    if not run_dir.is_dir():
        raise HTTPException(status_code=404, detail="Run not found")
    return run_dir


def read_run_json(run_dir: Path, filename: str, default: object) -> object:
    path = run_dir / filename
    if not path.is_file():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise HTTPException(status_code=500, detail=f"Run artifact {filename} is unreadable") from error


def run_process(run_config: object) -> str:
    """Read the process from run metadata; older configs are discrete DDPMs."""
    if not isinstance(run_config, dict):
        return "ddpm"
    config = run_config.get("config", {})
    if not isinstance(config, dict):
        return "ddpm"
    process = config.get("process", "ddpm")
    return process if isinstance(process, str) else "ddpm"


@lru_cache(maxsize=1)
def validation_images() -> UnconditionalEuroSATSplit:
    dataset = load_eurosat_rgb(PROJECT_ROOT / DATASET_CONFIG["cache_dir"])
    return UnconditionalEuroSATSplit(dataset[DATASET_CONFIG["validation_split"]])


def encode_normalized_image(image: torch.Tensor) -> bytes:
    pixels = (
        image.detach()
        .add(1.0)
        .mul(127.5)
        .round()
        .clamp(0, 255)
        .to(torch.uint8)
        .permute(1, 2, 0)
        .cpu()
        .numpy()
    )
    output = io.BytesIO()
    Image.fromarray(pixels).save(output, format="PNG")
    return output.getvalue()


def encode_pil_image(image: Image.Image) -> bytes:
    output = io.BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


@dataclass
class ActiveTrace:
    trace_id: str
    image_index: int
    mode: Literal["direct", "markov"]
    seed: int
    x0: torch.Tensor
    direct_noise: torch.Tensor | None
    states: torch.Tensor | None


_active_trace: ActiveTrace | None = None
_trace_lock = threading.Lock()
_generation_lock = threading.Lock()
_generation_progress_lock = threading.Lock()
_active_sampling_job: tuple[str, str] | None = None
_active_judgement_lock = threading.Lock()
_active_judgements: set[tuple[str, str]] = set()


class CreateTraceRequest(BaseModel):
    image_index: int = Field(ge=0)
    mode: Literal["direct", "markov"]
    seed: int = Field(ge=0, le=4_294_967_295)


class CreateGenerationRequest(BaseModel):
    num_samples: int = Field(default=8, ge=1, le=8)
    seed: int = Field(default=42, ge=0, le=4_294_967_295)
    checkpoint: str = Field(default="best-val.pt", min_length=1, max_length=100)
    sampler: Literal["reverse_sde", "annealed_langevin"] = "reverse_sde"


class CreateGenerationStudyRequest(BaseModel):
    num_samples: int = Field(default=100, ge=1, le=1000)
    seed: int = Field(default=42, ge=0, le=4_294_967_295)
    checkpoint: str = Field(default="best-val-ema.pt", min_length=1, max_length=100)


def atomic_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    for attempt in range(8):
        try:
            os.replace(temporary, path)
            break
        except PermissionError:
            if attempt == 7:
                raise
            time.sleep(0.015 * (attempt + 1))


def read_generation_progress(output_dir: Path) -> dict[str, object]:
    with _generation_progress_lock:
        value = read_run_json(output_dir, "progress.json", {})
    return value if isinstance(value, dict) else {}


def write_generation_progress(path: Path, value: dict[str, object]) -> None:
    with _generation_progress_lock:
        atomic_json(path, value)


def generation_dir(run_dir: Path, generation_id: str) -> Path:
    if not re.fullmatch(r"[0-9a-f]{32}", generation_id):
        raise HTTPException(status_code=404, detail="Generation not found")
    path = run_dir / "generation" / generation_id
    if not path.is_dir():
        raise HTTPException(status_code=404, detail="Generation not found")
    return path


def run_generation_job(
    run_id: str,
    generation_id: str,
    checkpoint_path: Path,
    output_dir: Path,
    num_samples: int,
    seed: int,
    checkpoint_name: str,
    process: str,
    sampler: str = "reverse_sde",
) -> None:
    global _active_sampling_job
    progress_path = output_dir / "progress.json"
    try:
        progress = read_generation_progress(output_dir)
        if not isinstance(progress, dict):
            progress = {}
        progress.update(state="running", updated_at=datetime.now(UTC).isoformat())
        progress["checkpoint"] = checkpoint_name
        write_generation_progress(progress_path, progress)

        def publish_ddpm_frame(frame_index: int, timestep: int, image: torch.Tensor) -> None:
            frames_dir = output_dir / "frames"
            frames_dir.mkdir(parents=True, exist_ok=True)
            frame_path = frames_dir / f"frame-{frame_index:05d}.png"
            display_grid(image.unsqueeze(0), frame_path, columns=1)
            progress.update(
                state="running",
                process="ddpm",
                completed_steps=min(frame_index * LIVE_FRAME_STRIDE, SCHEDULE.timesteps),
                total_steps=SCHEDULE.timesteps,
                current_timestep=timestep,
                latest_frame_index=frame_index,
                updated_at=datetime.now(UTC).isoformat(),
            )
            write_generation_progress(progress_path, progress)

        def publish_sde_frame(frame_index: int, step: int, current_time: float, image: torch.Tensor) -> None:
            frames_dir = output_dir / "frames"
            frames_dir.mkdir(parents=True, exist_ok=True)
            frame_path = frames_dir / f"frame-{frame_index:05d}.png"
            display_grid(image.unsqueeze(0), frame_path, columns=1)
            progress.update(
                state="running",
                process="vp_sde",
                completed_steps=step,
                total_steps=int(progress.get("total_steps", 1000)),
                current_t=current_time,
                latest_frame_index=frame_index,
                updated_at=datetime.now(UTC).isoformat(),
            )
            write_generation_progress(progress_path, progress)

        device = choose_device("auto")
        reference_count = min(128, len(validation_images()))
        reference = torch.stack([validation_images()[index] for index in range(reference_count)])
        if process == "vp_sde":
            run_config = read_run_json(get_run_dir(run_id), "config.json", {})
            wrapped = run_config.get("config", {}) if isinstance(run_config, dict) else {}
            sde_steps = int(wrapped.get("vp_sde", {}).get("sampling_steps", 1000)) if isinstance(wrapped, dict) else 1000
            progress["total_steps"] = 1000 if sampler == "annealed_langevin" else sde_steps
            progress["sampler"] = sampler
            write_generation_progress(progress_path, progress)
            metadata = generate_from_vp_sde_checkpoint(
                checkpoint_path, output_dir, num_samples=num_samples, seed=seed,
                device_name=str(device), on_frame=publish_sde_frame,
                validation_reference_images=reference,
                validation_reference_selection="first 128 validation images by split index (or full split if smaller)",
                sampling_steps=sde_steps,
                sampler=sampler,
            )
            schedule_steps = 1000 if sampler == "annealed_langevin" else sde_steps
            if sampler == "annealed_langevin":
                completed_time: float | None = float(wrapped.get("vp_sde", {}).get("training_time_min", 1e-5))
            else:
                completed_time = 0.0
        else:
            metadata = generate_from_checkpoint(
                checkpoint_path, output_dir, num_samples=num_samples, seed=seed,
                device_name=str(device), on_frame=publish_ddpm_frame,
                validation_reference_images=reference,
            )
            schedule_steps = SCHEDULE.timesteps
            completed_time = None
        completed = {
            "generation_id": generation_id,
            "run_id": run_id,
            "state": "completed",
            "process": process,
            "sampler": sampler,
            "num_samples": num_samples,
            "seed": seed,
            "checkpoint": checkpoint_name,
            "total_steps": schedule_steps,
            "completed_steps": schedule_steps,
            "current_timestep": 0,
            "current_t": completed_time,
            "latest_frame_index": schedule_steps // LIVE_FRAME_STRIDE,
            "frame_stride": LIVE_FRAME_STRIDE,
            "updated_at": datetime.now(UTC).isoformat(),
            "metadata": metadata,
        }
        write_generation_progress(progress_path, completed)
    except Exception as error:
        failed = read_generation_progress(output_dir)
        failed.update(
            generation_id=generation_id,
            run_id=run_id,
            state="failed",
            error=f"{type(error).__name__}: {error}",
            updated_at=datetime.now(UTC).isoformat(),
        )
        write_generation_progress(progress_path, failed)
    finally:
        with _generation_lock:
            if _active_sampling_job == (run_id, generation_id):
                _active_sampling_job = None


def generation_study_dir(run_dir: Path, study_id: str) -> Path:
    if not re.fullmatch(r"[0-9a-f]{32}", study_id):
        raise HTTPException(status_code=404, detail="Generation study not found")
    path = run_dir / "generation-studies" / study_id
    if not path.is_dir():
        raise HTTPException(status_code=404, detail="Generation study not found")
    return path


def read_study_status(path: Path) -> dict[str, object]:
    value = read_run_json(path, "status.json", {})
    return value if isinstance(value, dict) else {}


def update_study_status(path: Path, **updates: object) -> dict[str, object]:
    value = read_study_status(path)
    value.update(updates, updated_at=datetime.now(UTC).isoformat())
    atomic_json(path / "status.json", value)
    return value


def run_generation_study_job(
    run_id: str,
    study_id: str,
    checkpoint_path: Path,
    output_dir: Path,
    count: int,
    seed: int,
    checkpoint_name: str,
) -> None:
    global _active_sampling_job
    try:
        update_study_status(
            output_dir,
            generation_state="running",
            generation_completed=0,
            generation_total=count,
        )
        reference_count = min(REFERENCE_COUNT, len(validation_images()))
        if reference_count != REFERENCE_COUNT:
            raise RuntimeError("The validation split does not contain the fixed 16-image reference set")
        reference_indices = sorted(
            np.random.default_rng(42).choice(len(validation_images()), size=reference_count, replace=False).tolist()
        )
        reference = torch.stack([validation_images()[index] for index in reference_indices])

        def on_progress(completed: int, total: int) -> None:
            update_study_status(
                output_dir,
                generation_state="running",
                generation_completed=completed,
                generation_total=total,
            )

        metadata = generate_cohort(
            checkpoint_path,
            output_dir,
            count=count,
            seed=seed,
            validation_reference_images=reference,
            validation_reference_selection=f"16 unlabeled validation patches; fixed RNG seed 42; indices {reference_indices}",
            on_progress=on_progress,
        )
        update_study_status(
            output_dir,
            generation_state="completed",
            generation_completed=count,
            generation_total=count,
            judgement_state="not_started",
            judgement_completed=0,
            judgement_total=count,
            checkpoint=checkpoint_name,
            sample_statistics=metadata.get("sample_statistics", {}),
            sample_value_range=metadata.get("sample_value_range", []),
            fraction_outside_training_range=metadata.get("fraction_outside_training_range", 0.0),
        )
    except Exception as error:
        update_study_status(
            output_dir,
            generation_state="failed",
            error=f"{type(error).__name__}: {error}",
        )
    finally:
        with _generation_lock:
            if _active_sampling_job == (run_id, study_id):
                _active_sampling_job = None


def run_judgement_job(run_id: str, study_id: str, output_dir: Path) -> None:
    key = (run_id, study_id)
    try:
        update_study_status(output_dir, judgement_state="running", judgement_error=None)
        sample_path = output_dir / "samples.pt"
        if not sample_path.is_file():
            raise RuntimeError("Saved patch tensors are missing")
        samples = torch.load(sample_path, map_location="cpu", weights_only=True)
        metadata = read_run_json(output_dir, "metadata.json", {})
        if not isinstance(metadata, dict):
            raise RuntimeError("Generation metadata is unreadable")
        reference_path = output_dir / "reference.png"
        if not reference_path.is_file():
            raise RuntimeError("The fixed EuroSAT validation reference image is missing")
        reference_sheet = reference_path.read_bytes()

        judgement_path = output_dir / "judgements.json"
        stored = read_run_json(output_dir, "judgements.json", {})
        if not isinstance(stored, dict):
            stored = {}
        items = stored.get("items", [])
        batches = stored.get("batches", [])
        usage = stored.get("usage", {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0})
        results_by_id = {
            item["patch_id"]: item
            for item in items
            if isinstance(item, dict) and isinstance(item.get("patch_id"), str)
        } if isinstance(items, list) else {}
        if not isinstance(batches, list):
            batches = []
        if not isinstance(usage, dict):
            usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

        client = OpenAI()
        count = int(samples.shape[0])
        total_batches = (count + JUDGEMENT_BATCH_SIZE - 1) // JUDGEMENT_BATCH_SIZE
        completed_batches = {
            int(item["batch_index"])
            for item in batches
            if isinstance(item, dict) and isinstance(item.get("batch_index"), int)
        }
        update_study_status(
            output_dir,
            judgement_state="running",
            judgement_completed=len(results_by_id),
            judgement_total=count,
            judgement_completed_batches=len(completed_batches),
            judgement_total_batches=total_batches,
        )

        for batch_index, start in enumerate(range(0, count, JUDGEMENT_BATCH_SIZE)):
            end = min(start + JUDGEMENT_BATCH_SIZE, count)
            expected_ids = [f"patch-{index + 1:04d}" for index in range(start, end)]
            if all(patch_id in results_by_id for patch_id in expected_ids):
                continue
            update_study_status(
                output_dir,
                judgement_state="running",
                judgement_current_batch=batch_index + 1,
                judgement_total_batches=total_batches,
            )
            batch_results, batch_usage = judge_batch(
                client,
                patch_ids=expected_ids,
                images=samples[start:end],
                reference_sheet=reference_sheet,
            )
            for result in batch_results:
                results_by_id[result["patch_id"]] = result
            if batch_index not in completed_batches:
                batches.append({"batch_index": batch_index, "patch_ids": expected_ids})
                completed_batches.add(batch_index)
            for name in ("input_tokens", "output_tokens", "total_tokens"):
                usage[name] = int(usage.get(name, 0)) + int(batch_usage[name])
            ordered = [results_by_id[f"patch-{index + 1:04d}"] for index in range(count) if f"patch-{index + 1:04d}" in results_by_id]
            atomic_json(judgement_path, {"items": ordered, "batches": batches, "usage": usage})
            update_study_status(
                output_dir,
                judgement_state="running",
                judgement_completed=len(results_by_id),
                judgement_completed_batches=len(completed_batches),
                judgement_total_batches=total_batches,
            )

        ordered = [results_by_id[f"patch-{index + 1:04d}"] for index in range(count)]
        report = build_judgement_report(ordered, metadata)
        report["usage"] = usage
        atomic_json(output_dir / "judgement-report.json", report)
        update_study_status(
            output_dir,
            judgement_state="completed",
            judgement_completed=count,
            judgement_completed_batches=total_batches,
            judgement_total_batches=total_batches,
        )
    except Exception as error:
        update_study_status(
            output_dir,
            judgement_state="failed",
            judgement_error=f"Judging stopped ({type(error).__name__}). Completed batches are saved; retry to resume.",
        )
    finally:
        with _active_judgement_lock:
            _active_judgements.discard(key)


@app.get("/api/health")
def health() -> dict[str, str | bool]:
    trained = False
    if RUNS_ROOT.exists():
        for path in RUNS_ROOT.iterdir():
            if not path.is_dir():
                continue
            status = read_run_json(path, "status.json", {})
            if (
                isinstance(status, dict)
                and status.get("state") == "completed"
                and status.get("mode") == "train"
                and (path / "checkpoints" / "best-val.pt").is_file()
            ):
                trained = True
                break
    return {"status": "ok", "dataset_ready": True, "model_trained": trained}


@app.get("/api/runs")
def list_runs() -> dict[str, list[dict[str, object]]]:
    items: list[dict[str, object]] = []
    if RUNS_ROOT.is_dir():
        for run_dir in RUNS_ROOT.iterdir():
            if not run_dir.is_dir():
                continue
            status = read_run_json(run_dir, "status.json", {})
            run_config = read_run_json(run_dir, "config.json", {})
            if not isinstance(status, dict) or not isinstance(run_config, dict):
                continue
            process = run_process(run_config)
            generation_checkpoint = "best-val-ema.pt" if process == "vp_sde" else "best-val.pt"
            items.append(
                {
                    "run_id": run_dir.name,
                    "state": status.get("state", "unknown"),
                    "mode": status.get("mode", "train"),
                    "process": process,
                    "prediction": (
                        run_config.get("config", {}).get("model", {}).get("prediction")
                        if isinstance(run_config.get("config"), dict)
                        and isinstance(run_config.get("config", {}).get("model"), dict)
                        else "epsilon"
                    ),
                    "global_step": status.get("global_step", 0),
                    "epoch": status.get("epoch", 0),
                    "total_epochs": status.get("total_epochs", 0),
                    "created_at": run_config.get("created_at"),
                    "updated_at": status.get("updated_at"),
                    "can_generate": (
                        status.get("state") == "completed"
                        and status.get("mode") == "train"
                        and (run_dir / "checkpoints" / generation_checkpoint).is_file()
                    ),
                }
            )
    items.sort(key=lambda item: str(item.get("created_at") or ""), reverse=True)
    return {"items": items}


@app.get("/api/runs/{run_id}")
def get_run(run_id: str) -> dict[str, object]:
    run_dir = get_run_dir(run_id)
    return {
        "run_id": run_id,
        "status": read_run_json(run_dir, "status.json", {}),
        "run": read_run_json(run_dir, "config.json", {}),
    }


@app.get("/api/runs/{run_id}/metrics")
def get_run_metrics(
    run_id: str,
    after_event: int = Query(default=0, ge=0),
) -> dict[str, object]:
    run_dir = get_run_dir(run_id)
    monitor_path = run_dir / "monitor_metrics.jsonl"
    path = monitor_path if monitor_path.is_file() else run_dir / "metrics.jsonl"
    events: list[dict[str, object]] = []
    if path.is_file():
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if int(event.get("event_id", 0)) > after_event:
                    events.append(event)
    return {"items": events}


@app.get("/api/runs/{run_id}/checkpoints")
def get_run_checkpoints(run_id: str) -> dict[str, object]:
    run_dir = get_run_dir(run_id)
    manifest = read_run_json(run_dir, "checkpoint_manifest.json", [])
    if not isinstance(manifest, list):
        return {"items": []}
    items = [
        item
        for item in manifest
        if isinstance(item, dict)
        and isinstance(item.get("name"), str)
        and (run_dir / "checkpoints" / item["name"]).is_file()
    ]
    return {"items": items}


@app.post("/api/runs/{run_id}/generations", status_code=202)
def create_generation(
    run_id: str,
    request: CreateGenerationRequest,
    background_tasks: BackgroundTasks,
) -> dict[str, str | int]:
    global _active_sampling_job
    run_dir = get_run_dir(run_id)
    run_config = read_run_json(run_dir, "config.json", {})
    process = run_process(run_config)
    saved_config = run_config.get("config", {}) if isinstance(run_config, dict) else {}
    process_steps = (
        int(saved_config.get("vp_sde", {}).get("sampling_steps", 1000))
        if process == "vp_sde" and isinstance(saved_config, dict)
        else SCHEDULE.timesteps
    )
    if process == "vp_sde" and request.sampler == "annealed_langevin":
        process_steps = 1000  # 100 noise levels x 10 Langevin updates per level
    elif process != "vp_sde" and request.sampler != "reverse_sde":
        raise HTTPException(status_code=422, detail="Annealed Langevin is available only for VP-SDE runs")
    status = read_run_json(run_dir, "status.json", {})
    manifest = read_run_json(run_dir, "checkpoint_manifest.json", [])
    available = {
        item.get("name")
        for item in manifest
        if isinstance(item, dict) and isinstance(item.get("name"), str)
    } if isinstance(manifest, list) else set()
    if request.checkpoint not in available or Path(request.checkpoint).name != request.checkpoint:
        raise HTTPException(status_code=404, detail="Checkpoint not found in this run")
    checkpoint_path = run_dir / "checkpoints" / request.checkpoint
    if (
        not isinstance(status, dict)
        or status.get("state") != "completed"
        or status.get("mode") != "train"
        or not checkpoint_path.is_file()
    ):
        raise HTTPException(status_code=409, detail="This run has no completed best-validation checkpoint")

    with _generation_lock:
        if _active_sampling_job is not None:
            raise HTTPException(status_code=409, detail="A generation is already running")
        generation_id = uuid4().hex
        output_dir = run_dir / "generation" / generation_id
        output_dir.mkdir(parents=True, exist_ok=False)
        initial = {
            "generation_id": generation_id,
            "run_id": run_id,
            "state": "queued",
            "process": process,
            "sampler": request.sampler,
            "num_samples": request.num_samples,
            "seed": request.seed,
            "checkpoint": request.checkpoint,
            "total_steps": process_steps,
            "completed_steps": 0,
            "current_timestep": SCHEDULE.timesteps,
            "current_t": 1.0 if process == "vp_sde" else None,
            "latest_frame_index": -1,
            "frame_stride": LIVE_FRAME_STRIDE,
        }
        write_generation_progress(output_dir / "progress.json", initial)
        _active_sampling_job = (run_id, generation_id)
        background_tasks.add_task(
            run_generation_job,
            run_id,
            generation_id,
            checkpoint_path,
            output_dir,
            request.num_samples,
            request.seed,
            request.checkpoint,
            process,
            request.sampler,
        )
    return {"generation_id": generation_id, "run_id": run_id, "state": "queued"}


@app.get("/api/runs/{run_id}/generations/{generation_id}")
def get_generation(run_id: str, generation_id: str) -> dict[str, object]:
    run_dir = get_run_dir(run_id)
    output_dir = generation_dir(run_dir, generation_id)
    progress = read_generation_progress(output_dir)
    if progress.get("state") in {"queued", "running"}:
        with _generation_lock:
            is_active = _active_sampling_job == (run_id, generation_id)
        if not is_active:
            progress.update(state="interrupted", error="The API stopped before generation completed")
    return progress


@app.get("/api/runs/{run_id}/generation-studies")
def list_generation_studies(run_id: str) -> dict[str, object]:
    run_dir = get_run_dir(run_id)
    root = run_dir / "generation-studies"
    items: list[dict[str, object]] = []
    if root.is_dir():
        for path in root.iterdir():
            if not path.is_dir():
                continue
            status = read_study_status(path)
            items.append({
                "study_id": path.name,
                "run_id": run_id,
                "created_at": status.get("created_at"),
                "generation_state": status.get("generation_state"),
                "generation_completed": status.get("generation_completed", 0),
                "generation_total": status.get("generation_total", 0),
                "judgement_state": status.get("judgement_state"),
                "judgement_completed": status.get("judgement_completed", 0),
                "judgement_total": status.get("judgement_total", 0),
                "num_samples": status.get("num_samples"),
                "seed": status.get("seed"),
                "checkpoint": status.get("checkpoint"),
                "can_judge": bool(os.environ.get("OPENAI_API_KEY")),
            })
    items.sort(key=lambda item: str(item.get("created_at", "")), reverse=True)
    return {"items": items}


@app.post("/api/runs/{run_id}/generation-studies", status_code=202)
def create_generation_study(
    run_id: str,
    request: CreateGenerationStudyRequest,
    background_tasks: BackgroundTasks,
) -> dict[str, str | int]:
    global _active_sampling_job
    run_dir = get_run_dir(run_id)
    if run_process(read_run_json(run_dir, "config.json", {})) == "vp_sde":
        raise HTTPException(
            status_code=409,
            detail="VP-SDE cohort generation and VLM judgment are not enabled in this workflow.",
        )
    status = read_run_json(run_dir, "status.json", {})
    manifest = read_run_json(run_dir, "checkpoint_manifest.json", [])
    available = {
        item.get("name")
        for item in manifest
        if isinstance(item, dict) and isinstance(item.get("name"), str)
    } if isinstance(manifest, list) else set()
    if request.checkpoint not in available or Path(request.checkpoint).name != request.checkpoint:
        raise HTTPException(status_code=404, detail="Checkpoint not found in this run")
    checkpoint_path = run_dir / "checkpoints" / request.checkpoint
    if (
        not isinstance(status, dict)
        or status.get("state") != "completed"
        or status.get("mode") != "train"
        or not checkpoint_path.is_file()
    ):
        raise HTTPException(status_code=409, detail="This run has no completed training checkpoint")

    with _generation_lock:
        if _active_sampling_job is not None:
            raise HTTPException(status_code=409, detail="Another sampling job is already running")
        study_id = uuid4().hex
        output_dir = run_dir / "generation-studies" / study_id
        output_dir.mkdir(parents=True, exist_ok=False)
        initial = {
            "run_id": run_id,
            "created_at": datetime.now(UTC).isoformat(),
            "generation_state": "queued",
            "generation_completed": 0,
            "generation_total": request.num_samples,
            "judgement_state": "not_started",
            "judgement_completed": 0,
            "judgement_total": request.num_samples,
            "judgement_batch_size": JUDGEMENT_BATCH_SIZE,
            "generation_chunk_size": GENERATION_CHUNK_SIZE,
            "seed": request.seed,
            "num_samples": request.num_samples,
            "checkpoint": request.checkpoint,
            "can_judge": bool(os.environ.get("OPENAI_API_KEY")),
            "updated_at": datetime.now(UTC).isoformat(),
        }
        atomic_json(output_dir / "status.json", initial)
        _active_sampling_job = (run_id, study_id)
        background_tasks.add_task(
            run_generation_study_job,
            run_id,
            study_id,
            checkpoint_path,
            output_dir,
            request.num_samples,
            request.seed,
            request.checkpoint,
        )
    return {"study_id": study_id, "run_id": run_id, "generation_state": "queued"}


@app.get("/api/runs/{run_id}/generation-studies/{study_id}")
def get_generation_study(run_id: str, study_id: str) -> dict[str, object]:
    run_dir = get_run_dir(run_id)
    output_dir = generation_study_dir(run_dir, study_id)
    status = read_study_status(output_dir)
    if status.get("generation_state") in {"queued", "running"}:
        with _generation_lock:
            is_active = _active_sampling_job == (run_id, study_id)
        if not is_active:
            status = update_study_status(
                output_dir,
                generation_state="interrupted",
                error="The API stopped before cohort generation completed",
            )
    if status.get("judgement_state") in {"queued", "running"}:
        with _active_judgement_lock:
            is_active = (run_id, study_id) in _active_judgements
        if not is_active:
            status = update_study_status(
                output_dir,
                judgement_state="failed",
                judgement_error="Judging was interrupted; retry to resume saved batches.",
            )
    status["study_id"] = study_id
    status["can_judge"] = bool(os.environ.get("OPENAI_API_KEY"))
    return status


@app.get("/api/runs/{run_id}/generation-studies/{study_id}/samples.png")
def get_generation_study_grid(run_id: str, study_id: str) -> FileResponse:
    run_dir = get_run_dir(run_id)
    output_dir = generation_study_dir(run_dir, study_id)
    status = read_study_status(output_dir)
    if status.get("generation_state") != "completed":
        raise HTTPException(status_code=409, detail="The generated cohort is not ready")
    path = output_dir / "samples.png"
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Generated cohort grid not found")
    return FileResponse(path, media_type="image/png", headers={"Cache-Control": "no-store"})


@app.get("/api/runs/{run_id}/generation-studies/{study_id}/samples/{sample_index}.png")
def get_generation_study_patch(run_id: str, study_id: str, sample_index: int) -> FileResponse:
    run_dir = get_run_dir(run_id)
    output_dir = generation_study_dir(run_dir, study_id)
    status = read_study_status(output_dir)
    count = int(status.get("num_samples", 0))
    if status.get("generation_state") != "completed" or not 0 <= sample_index < count:
        raise HTTPException(status_code=404, detail="Generated patch not found")
    path = output_dir / "patches" / f"patch-{sample_index + 1:04d}.png"
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Generated patch not found")
    return FileResponse(path, media_type="image/png", headers={"Cache-Control": "no-store"})


@app.post("/api/runs/{run_id}/generation-studies/{study_id}/judgements", status_code=202)
def start_generation_study_judgement(
    run_id: str,
    study_id: str,
    background_tasks: BackgroundTasks,
) -> dict[str, str]:
    run_dir = get_run_dir(run_id)
    output_dir = generation_study_dir(run_dir, study_id)
    status = read_study_status(output_dir)
    if status.get("generation_state") != "completed":
        raise HTTPException(status_code=409, detail="Generate and review the cohort before judging it")
    if not os.environ.get("OPENAI_API_KEY"):
        raise HTTPException(status_code=503, detail="OPENAI_API_KEY is not available to the API process")
    if status.get("judgement_state") == "completed":
        raise HTTPException(status_code=409, detail="This cohort has already been judged")

    key = (run_id, study_id)
    with _active_judgement_lock:
        if _active_judgements:
            if key in _active_judgements:
                raise HTTPException(status_code=409, detail="Judging is already running for this cohort")
            raise HTTPException(status_code=409, detail="Another cohort is currently being judged")
        _active_judgements.add(key)
    update_study_status(output_dir, judgement_state="queued", judgement_error=None)
    background_tasks.add_task(run_judgement_job, run_id, study_id, output_dir)
    return {"study_id": study_id, "judgement_state": "queued"}


@app.get("/api/runs/{run_id}/generation-studies/{study_id}/judgements")
def get_generation_study_judgements(run_id: str, study_id: str) -> dict[str, object]:
    run_dir = get_run_dir(run_id)
    output_dir = generation_study_dir(run_dir, study_id)
    values = read_run_json(output_dir, "judgements.json", {})
    report = read_run_json(output_dir, "judgement-report.json", {})
    return {
        "items": values.get("items", []) if isinstance(values, dict) else [],
        "report": report if isinstance(report, dict) else {},
    }


@app.get("/api/runs/{run_id}/generations/{generation_id}/frames/{frame_index}")
def get_generation_frame(run_id: str, generation_id: str, frame_index: int) -> Response:
    run_dir = get_run_dir(run_id)
    output_dir = generation_dir(run_dir, generation_id)
    progress = read_generation_progress(output_dir)
    latest = int(progress.get("latest_frame_index", -1))
    if frame_index < 0 or frame_index > latest:
        raise HTTPException(status_code=404, detail="That denoising frame is not available yet")
    frame_path = output_dir / "frames" / f"frame-{frame_index:05d}.png"
    if not frame_path.is_file():
        raise HTTPException(status_code=404, detail="Denoising frame not found")
    process = str(progress.get("process", run_process(read_run_json(run_dir, "config.json", {}))))
    headers = {"X-Process": process, "Cache-Control": "no-store"}
    if process == "vp_sde":
        total_steps = max(int(progress.get("total_steps", 1000)), 1)
        if progress.get("sampler") == "annealed_langevin":
            run_config = read_run_json(run_dir, "config.json", {})
            wrapped = run_config.get("config", {}) if isinstance(run_config, dict) else {}
            time_min = float(wrapped.get("vp_sde", {}).get("training_time_min", 1e-5)) if isinstance(wrapped, dict) else 1e-5
            current_time = (
                1.0 if frame_index == 0
                else 1.0 - ((frame_index - 1) / max(total_steps // LIVE_FRAME_STRIDE - 1, 1)) * (1.0 - time_min)
            )
        else:
            current_time = max(1.0 - frame_index * LIVE_FRAME_STRIDE / total_steps, 0.0)
        headers["X-Time"] = f"{current_time:.8f}"
    else:
        timestep = max(SCHEDULE.timesteps - frame_index * LIVE_FRAME_STRIDE, 0)
        headers["X-Timestep"] = str(timestep)
    return FileResponse(
        frame_path,
        media_type="image/png",
        headers=headers,
    )


@app.get("/api/runs/{run_id}/generations/{generation_id}/samples.png")
def get_generation_samples(run_id: str, generation_id: str) -> FileResponse:
    run_dir = get_run_dir(run_id)
    output_dir = generation_dir(run_dir, generation_id)
    progress = read_generation_progress(output_dir)
    if progress.get("state") != "completed":
        raise HTTPException(status_code=409, detail="Generated sample grid is not ready")
    samples_path = output_dir / "samples.png"
    if not samples_path.is_file():
        raise HTTPException(status_code=404, detail="Generated sample grid not found")
    return FileResponse(samples_path, media_type="image/png", headers={"Cache-Control": "no-store"})


@app.get("/api/learning-log")
def learning_log() -> FileResponse:
    return FileResponse(PROJECT_ROOT / "docs" / "learning-log.md", media_type="text/markdown")


@app.get("/api/validation/images")
def list_validation_images(
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=24, ge=1, le=48),
) -> dict[str, object]:
    images = validation_images()
    end = min(offset + limit, len(images))
    items = [
        {"index": index, "url": f"/api/validation/images/{index}"}
        for index in range(offset, end)
    ]
    return {"total": len(images), "offset": offset, "limit": limit, "items": items}


@app.get("/api/validation/images/{image_index}")
def get_validation_image(image_index: int) -> Response:
    images = validation_images()
    if image_index < 0 or image_index >= len(images):
        raise HTTPException(status_code=404, detail="Validation image not found")
    # The adapter selected only the image column, so labels and filenames are absent.
    image = images.images[image_index]["image"]
    return Response(encode_pil_image(image), media_type="image/png")


@app.post("/api/forward-traces", status_code=201)
def create_forward_trace(request: CreateTraceRequest) -> dict[str, int | str]:
    images = validation_images()
    if request.image_index >= len(images):
        raise HTTPException(status_code=404, detail="Validation image not found")

    x0 = images[request.image_index].unsqueeze(0)
    if request.mode == "direct":
        generator = torch.Generator(device=x0.device).manual_seed(request.seed)
        direct_noise = torch.randn(
            x0.shape,
            generator=generator,
            device=x0.device,
            dtype=x0.dtype,
        )
        states = None
    else:
        direct_noise = None
        states = q_sample_markov(SCHEDULE, x0, request.seed)

    trace_id = uuid4().hex
    trace = ActiveTrace(
        trace_id=trace_id,
        image_index=request.image_index,
        mode=request.mode,
        seed=request.seed,
        x0=x0,
        direct_noise=direct_noise,
        states=states,
    )
    global _active_trace
    with _trace_lock:
        _active_trace = trace

    return {
        "trace_id": trace_id,
        "mode": request.mode,
        "steps": SCHEDULE.timesteps,
        "image_index": request.image_index,
        "seed": request.seed,
    }


@app.get("/api/forward-traces/{trace_id}/frames/{step_index}")
def get_trace_frame(trace_id: str, step_index: int) -> Response:
    if step_index < 0 or step_index >= SCHEDULE.timesteps:
        raise HTTPException(
            status_code=422,
            detail=f"step_index must be between 0 and {SCHEDULE.timesteps - 1}",
        )
    with _trace_lock:
        trace = _active_trace
    if trace is None or trace.trace_id != trace_id:
        raise HTTPException(status_code=404, detail="Trace not found or replaced")

    if trace.mode == "direct":
        timestep = torch.tensor([step_index], dtype=torch.long)
        frame, _ = q_sample(SCHEDULE, trace.x0, timestep, trace.direct_noise)
        frame = frame[0]
    else:
        assert trace.states is not None
        frame = trace.states[step_index, 0]

    alpha_bar = float(SCHEDULE.alpha_bars[step_index].item())
    return Response(
        encode_normalized_image(frame),
        media_type="image/png",
        headers={
            "X-Timestep-Index": str(step_index),
            "X-Timestep": str(step_index + 1),
            "X-Alpha-Bar": str(alpha_bar),
            "Cache-Control": "no-store",
        },
    )


frontend_dist = PROJECT_ROOT / "web" / "dist"
if (frontend_dist / "index.html").is_file():
    app.frontend("/", directory=str(frontend_dist))
