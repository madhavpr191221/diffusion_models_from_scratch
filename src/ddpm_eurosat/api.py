"""Local API for the pre-training forward-noise viewer."""

from __future__ import annotations

import io
import json
import re
import threading
import tomllib
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Literal
from uuid import uuid4

import numpy as np
import torch
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, Field
from PIL import Image

from ddpm_eurosat.data import UnconditionalEuroSATSplit, image_to_tensor, load_eurosat_rgb
from ddpm_eurosat.diffusion import DiffusionSchedule, q_sample, q_sample_markov

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


class CreateTraceRequest(BaseModel):
    image_index: int = Field(ge=0)
    mode: Literal["direct", "markov"]
    seed: int = Field(ge=0, le=4_294_967_295)


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
    path = run_dir / "metrics.jsonl"
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
    return {"items": read_run_json(run_dir, "checkpoint_manifest.json", [])}


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
