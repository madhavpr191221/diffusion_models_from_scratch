"""Generate unconditional samples with the ancestral DDPM reverse process."""

from __future__ import annotations

import argparse
import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from ddpm_eurosat.diffusion import DiffusionSchedule, ddpm_reverse_step
from ddpm_eurosat.model import TimeConditionedUNet

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return torch.device(requested)


def display_grid(images: torch.Tensor, path: Path, columns: int) -> None:
    """Save normalized RGB tensors as a simple, labeled-grid-free PNG."""
    pixels = ((images.detach().cpu().clamp(-1, 1) + 1.0) * 127.5)
    pixels = pixels.to(torch.uint8).permute(0, 2, 3, 1).numpy()
    count, height, width, _ = pixels.shape
    rows = (count + columns - 1) // columns
    canvas = np.zeros((rows * height, columns * width, 3), dtype=np.uint8)
    for index, image in enumerate(pixels):
        row, column = divmod(index, columns)
        canvas[row * height : (row + 1) * height, column * width : (column + 1) * width] = image
    Image.fromarray(canvas, mode="RGB").save(path)


def sample_chain(
    model: TimeConditionedUNet,
    schedule: DiffusionSchedule,
    *,
    count: int,
    seed: int,
    device: torch.device,
    trajectory_index: int | None = 0,
    progress: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Draw ``count`` samples and optionally retain one complete x_T..x_0 path."""
    generator = torch.Generator(device=device).manual_seed(seed)
    shape = (count, 3, 64, 64)
    current = torch.randn(shape, generator=generator, device=device, dtype=torch.float32)
    trajectory = [current[trajectory_index].detach().cpu()] if trajectory_index is not None else None
    model.eval()
    started = time.perf_counter()
    with torch.inference_mode():
        for timestep in reversed(range(schedule.timesteps)):
            t = torch.full((count,), timestep, dtype=torch.long, device=device)
            prediction = model(current, t)
            if timestep > 0:
                noise = torch.randn(current.shape, generator=generator, device=device, dtype=current.dtype)
            else:
                noise = None
            current = ddpm_reverse_step(schedule, current, prediction, timestep, noise=noise)
            if trajectory is not None:
                trajectory.append(current[trajectory_index].detach().cpu())
            if progress and ((schedule.timesteps - timestep) % 100 == 0 or timestep == 0):
                print(f"Reverse updates: {schedule.timesteps - timestep}/{schedule.timesteps}", flush=True)
    elapsed = time.perf_counter() - started
    print(f"Sampling completed in {elapsed:.1f}s ({count} samples, {schedule.timesteps} steps)", flush=True)
    return current.detach().cpu(), torch.stack(trajectory) if trajectory is not None else None


def load_checkpoint(path: Path, device: torch.device) -> tuple[dict[str, Any], dict[str, Any]]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint or "config" not in checkpoint:
        raise ValueError("Checkpoint does not contain the expected model state and resolved config")
    config = checkpoint["config"]
    diffusion_config = config["diffusion"]
    if diffusion_config.get("reverse_variance") != "posterior":
        raise ValueError("This sampler implements posterior reverse variance only")
    return checkpoint, config


def run(args: argparse.Namespace) -> Path:
    checkpoint_path = args.checkpoint.resolve()
    device = choose_device(args.device)
    checkpoint, config = load_checkpoint(checkpoint_path, device)
    diffusion = config["diffusion"]
    schedule = DiffusionSchedule(
        timesteps=int(diffusion["timesteps"]),
        beta_start=float(diffusion["beta_start"]),
        beta_end=float(diffusion["beta_end"]),
    ).to(device)
    model = TimeConditionedUNet().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    output_dir = (args.output_dir or checkpoint_path.parent.parent / "generation" / f"seed-{args.seed}").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    samples, trajectory = sample_chain(
        model,
        schedule,
        count=args.num_samples,
        seed=args.seed,
        device=device,
        trajectory_index=args.trajectory_index,
        progress=True,
    )
    if not torch.isfinite(samples).all():
        raise RuntimeError("Sampling produced non-finite image values")
    torch.save(samples, output_dir / "samples.pt")
    display_grid(samples, output_dir / "samples.png", columns=min(4, args.num_samples))

    if trajectory is not None:
        torch.save(trajectory, output_dir / "trajectory.pt")
        positions = torch.linspace(0, schedule.timesteps, steps=min(9, schedule.timesteps + 1)).round().long()
        display_grid(trajectory[positions], output_dir / "trajectory.png", columns=3)

    metadata = {
        "created_at": datetime.now(UTC).isoformat(),
        "checkpoint": str(checkpoint_path),
        "checkpoint_global_step": int(checkpoint.get("global_step", -1)),
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "device": str(device),
        "seed": args.seed,
        "num_samples": args.num_samples,
        "image_shape": [3, 64, 64],
        "timesteps": schedule.timesteps,
        "beta_start": float(diffusion["beta_start"]),
        "beta_end": float(diffusion["beta_end"]),
        "reverse_variance": "posterior",
        "trajectory_index": args.trajectory_index,
        "trajectory_frame_order": "x_T to x_0",
        "trajectory_frames": int(trajectory.shape[0]) if trajectory is not None else 0,
        "sample_value_range": [float(samples.min()), float(samples.max())],
        "display_clipping": "[-1,1] applied only when saving PNGs; raw tensors are preserved",
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, allow_nan=False), encoding="utf-8")
    print(f"Saved generation artifacts to {output_dir}", flush=True)
    return output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True, help="Training checkpoint, e.g. best-val.pt")
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trajectory-index", type=int, default=0, help="Sample index to retain, or -1 to disable")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    args = parser.parse_args()
    if args.num_samples < 1:
        parser.error("--num-samples must be at least 1")
    if args.trajectory_index == -1:
        args.trajectory_index = None
    elif not 0 <= args.trajectory_index < args.num_samples:
        parser.error("--trajectory-index must be -1 or in [0, num-samples)")
    return args


if __name__ == "__main__":
    run(parse_args())
