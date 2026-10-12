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

from ddpm_eurosat.data import UnconditionalEuroSATSplit, image_to_tensor, load_eurosat_rgb
from ddpm_eurosat.diffusion import DiffusionSchedule, q_sample, q_sample_markov
from ddpm_eurosat.sampling import LIVE_FRAME_STRIDE, choose_device, display_grid, generate_from_checkpoint

PROJECT_ROOT = Path(__file__).resolve().parents[2]
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
_active_generation: tuple[str, str] | None = None


class CreateTraceRequest(BaseModel):
    image_index: int = Field(ge=0)
    mode: Literal["direct", "markov"]
    seed: int = Field(ge=0, le=4_294_967_295)


class CreateGenerationRequest(BaseModel):
    num_samples: int = Field(default=8, ge=1, le=8)
    seed: int = Field(default=42, ge=0, le=4_294_967_295)
    checkpoint: str = Field(default="best-val.pt", min_length=1, max_length=100)


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
) -> None:
    global _active_generation
    progress_path = output_dir / "progress.json"
    try:
        progress = read_generation_progress(output_dir)
        if not isinstance(progress, dict):
            progress = {}
        progress.update(state="running", updated_at=datetime.now(UTC).isoformat())
        progress["checkpoint"] = checkpoint_name
        write_generation_progress(progress_path, progress)

        def publish_frame(frame_index: int, timestep: int, image: torch.Tensor) -> None:
            frames_dir = output_dir / "frames"
            frames_dir.mkdir(parents=True, exist_ok=True)
            frame_path = frames_dir / f"frame-{frame_index:05d}.png"
            display_grid(image.unsqueeze(0), frame_path, columns=1)
            progress.update(
                state="running",
                completed_steps=min(frame_index * LIVE_FRAME_STRIDE, SCHEDULE.timesteps),
                current_timestep=timestep,
                latest_frame_index=frame_index,
                updated_at=datetime.now(UTC).isoformat(),
            )
            write_generation_progress(progress_path, progress)

        schedule_steps = SCHEDULE.timesteps
        device = choose_device("auto")
        reference_count = min(128, len(validation_images()))
        reference = torch.stack([validation_images()[index] for index in range(reference_count)])
        metadata = generate_from_checkpoint(
            checkpoint_path,
            output_dir,
            num_samples=num_samples,
            seed=seed,
            device_name=str(device),
            on_frame=publish_frame,
            validation_reference_images=reference,
        )
        completed = {
            "generation_id": generation_id,
            "run_id": run_id,
            "state": "completed",
            "num_samples": num_samples,
            "seed": seed,
            "checkpoint": checkpoint_name,
            "total_steps": schedule_steps,
            "completed_steps": schedule_steps,
            "current_timestep": 0,
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
            if _active_generation == (run_id, generation_id):
                _active_generation = None


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
            items.append(
                {
                    "run_id": run_dir.name,
                    "state": status.get("state", "unknown"),
                    "mode": status.get("mode", "train"),
                    "global_step": status.get("global_step", 0),
                    "epoch": status.get("epoch", 0),
                    "total_epochs": status.get("total_epochs", 0),
                    "created_at": run_config.get("created_at"),
                    "updated_at": status.get("updated_at"),
                    "can_generate": (
                        status.get("state") == "completed"
                        and status.get("mode") == "train"
                        and (run_dir / "checkpoints" / "best-val.pt").is_file()
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
    global _active_generation
    run_dir = get_run_dir(run_id)
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
        if _active_generation is not None:
            raise HTTPException(status_code=409, detail="A generation is already running")
        generation_id = uuid4().hex
        output_dir = run_dir / "generation" / generation_id
        output_dir.mkdir(parents=True, exist_ok=False)
        initial = {
            "generation_id": generation_id,
            "run_id": run_id,
            "state": "queued",
            "num_samples": request.num_samples,
            "seed": request.seed,
            "checkpoint": request.checkpoint,
            "total_steps": SCHEDULE.timesteps,
            "completed_steps": 0,
            "current_timestep": SCHEDULE.timesteps,
            "latest_frame_index": -1,
            "frame_stride": LIVE_FRAME_STRIDE,
        }
        write_generation_progress(output_dir / "progress.json", initial)
        _active_generation = (run_id, generation_id)
        background_tasks.add_task(
            run_generation_job,
            run_id,
            generation_id,
            checkpoint_path,
            output_dir,
            request.num_samples,
            request.seed,
            request.checkpoint,
        )
    return {"generation_id": generation_id, "run_id": run_id, "state": "queued"}


@app.get("/api/runs/{run_id}/generations/{generation_id}")
def get_generation(run_id: str, generation_id: str) -> dict[str, object]:
    run_dir = get_run_dir(run_id)
    output_dir = generation_dir(run_dir, generation_id)
    progress = read_generation_progress(output_dir)
    if progress.get("state") in {"queued", "running"}:
        with _generation_lock:
            is_active = _active_generation == (run_id, generation_id)
        if not is_active:
            progress.update(state="interrupted", error="The API stopped before generation completed")
    return progress


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
    timestep = max(SCHEDULE.timesteps - frame_index * LIVE_FRAME_STRIDE, 0)
    return FileResponse(
        frame_path,
        media_type="image/png",
        headers={"X-Timestep": str(timestep), "Cache-Control": "no-store"},
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
