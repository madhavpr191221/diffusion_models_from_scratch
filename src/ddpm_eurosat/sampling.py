"""Generate unconditional samples with the ancestral DDPM reverse process."""

from __future__ import annotations

import argparse
import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from PIL import Image

from ddpm_eurosat.diffusion import DiffusionSchedule, ddpm_reverse_step
from ddpm_eurosat.model import build_model

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LIVE_FRAME_STRIDE = 10


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


def summarize_images(images: torch.Tensor) -> dict[str, Any]:
    """Return small, interpretable distribution and diversity summaries."""
    values = images.detach().to(device="cpu", dtype=torch.float32)
    if values.ndim != 4 or values.shape[1] != 3:
        raise ValueError("images must have shape [N,3,H,W]")
    if not torch.isfinite(values).all():
        raise ValueError("images must contain only finite values")
    flat = values.permute(1, 0, 2, 3).reshape(3, -1)
    channel_names = ("red", "green", "blue")
    channels = {
        name: {
            "mean": float(flat[index].mean()),
            "std": float(flat[index].std(unbiased=False)),
            "q01": float(torch.quantile(flat[index], 0.01)),
            "median": float(torch.quantile(flat[index], 0.50)),
            "q99": float(torch.quantile(flat[index], 0.99)),
        }
        for index, name in enumerate(channel_names)
    }
    image_pairs = values.flatten(1)
    pairs = [
        (left, right)
        for left in range(values.shape[0])
        for right in range(left + 1, values.shape[0])
    ]
    if len(pairs) > 256:
        selected = torch.linspace(0, len(pairs) - 1, steps=256).round().long().tolist()
        pairs = [pairs[index] for index in selected]
    pair_count = len(pairs)
    pairwise_rmse = 0.0
    if pair_count:
        pair_distances = [
            float(torch.mean((image_pairs[left] - image_pairs[right]).square()).sqrt())
            for left, right in pairs
        ]
        pairwise_rmse = sum(pair_distances) / pair_count
    return {
        "num_images": int(values.shape[0]),
        "value_range": [float(values.min()), float(values.max())],
        "fraction_outside_training_range": float((values.abs() > 1.0).float().mean()),
        "mean_per_image_pixel_std": float(values.flatten(1).std(dim=1, unbiased=False).mean()),
        "pairwise_pixel_rmse": pairwise_rmse if pair_count else None,
        "pair_count": pair_count,
        "channels": channels,
    }


def sample_chain(
    model: torch.nn.Module,
    schedule: DiffusionSchedule,
    *,
    count: int,
    seed: int,
    device: torch.device,
    trajectory_index: int | None = 0,
    variance_type: str = "posterior",
    progress: bool = False,
    on_frame: Callable[[int, int, torch.Tensor], None] | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Draw ``count`` samples and optionally retain one complete x_T..x_0 path."""
    generator = torch.Generator(device=device).manual_seed(seed)
    shape = (count, 3, 64, 64)
    current = torch.randn(shape, generator=generator, device=device, dtype=torch.float32)
    trajectory = [current[trajectory_index].detach().cpu()] if trajectory_index is not None else None
    if on_frame is not None and trajectory_index is not None:
        on_frame(0, schedule.timesteps, current[trajectory_index].detach().cpu())
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
            current = ddpm_reverse_step(
                schedule, current, prediction, timestep, noise=noise, variance_type=variance_type
            )
            if trajectory is not None:
                trajectory.append(current[trajectory_index].detach().cpu())
            updates_completed = schedule.timesteps - timestep
            if (
                on_frame is not None
                and trajectory_index is not None
                and (updates_completed % LIVE_FRAME_STRIDE == 0 or timestep == 0)
            ):
                on_frame(
                    updates_completed // LIVE_FRAME_STRIDE,
                    max(schedule.timesteps - updates_completed, 0),
                    current[trajectory_index].detach().cpu(),
                )
            if progress and ((schedule.timesteps - timestep) % 100 == 0 or timestep == 0):
                print(f"Reverse updates: {updates_completed}/{schedule.timesteps}", flush=True)
    elapsed = time.perf_counter() - started
    print(f"Sampling completed in {elapsed:.1f}s ({count} samples, {schedule.timesteps} steps)", flush=True)
    return current.detach().cpu(), torch.stack(trajectory) if trajectory is not None else None


def load_checkpoint(path: Path, device: torch.device) -> tuple[dict[str, Any], dict[str, Any]]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint or "config" not in checkpoint:
        raise ValueError("Checkpoint does not contain the expected model state and resolved config")
    config = checkpoint["config"]
    diffusion_config = config["diffusion"]
    if diffusion_config.get("reverse_variance", "posterior") not in {"posterior", "fixed_large"}:
        raise ValueError("Checkpoint requests an unsupported reverse variance")
    return checkpoint, config


def generate_from_checkpoint(
    checkpoint_path: Path,
    output_dir: Path,
    *,
    num_samples: int,
    seed: int,
    trajectory_index: int | None = 0,
    device_name: str = "auto",
    on_frame: Callable[[int, int, torch.Tensor], None] | None = None,
    progress: bool = False,
    validation_reference_images: torch.Tensor | None = None,
) -> dict[str, Any]:
    """Generate samples and persist artifacts using the shared DDPM sampler."""
    checkpoint_path = checkpoint_path.resolve()
    output_dir = output_dir.resolve()
    device = choose_device(device_name)
    checkpoint, config = load_checkpoint(checkpoint_path, device)
    diffusion = config["diffusion"]
    schedule = DiffusionSchedule(
        timesteps=int(diffusion["timesteps"]),
        beta_start=float(diffusion["beta_start"]),
        beta_end=float(diffusion["beta_end"]),
    ).to(device)
    model = build_model(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    output_dir.mkdir(parents=True, exist_ok=True)

    samples, trajectory = sample_chain(
        model,
        schedule,
        count=num_samples,
        seed=seed,
        device=device,
        trajectory_index=trajectory_index,
        variance_type=str(diffusion.get("reverse_variance", "posterior")),
        progress=progress,
        on_frame=on_frame,
    )
    if not torch.isfinite(samples).all():
        raise RuntimeError("Sampling produced non-finite image values")
    torch.save(samples, output_dir / "samples.pt")
    display_grid(samples, output_dir / "samples.png", columns=min(4, num_samples))

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
        "seed": seed,
        "num_samples": num_samples,
        "image_shape": [3, 64, 64],
        "timesteps": schedule.timesteps,
        "beta_start": float(diffusion["beta_start"]),
        "beta_end": float(diffusion["beta_end"]),
        "reverse_variance": str(diffusion.get("reverse_variance", "posterior")),
        "trajectory_index": trajectory_index,
        "trajectory_frame_order": "x_T to x_0",
        "trajectory_frames": int(trajectory.shape[0]) if trajectory is not None else 0,
        "sample_value_range": [float(samples.min()), float(samples.max())],
        "fraction_outside_training_range": float((samples.abs() > 1.0).float().mean()),
        "sample_statistics": summarize_images(samples),
        "per_sample_diagnostics": [
            {
                "minimum": float(sample.min()),
                "maximum": float(sample.max()),
                "mean": float(sample.mean()),
                "fraction_outside_training_range": float((sample.abs() > 1.0).float().mean()),
            }
            for sample in samples
        ],
        "validation_reference": (
            {
                "split": "validation",
                "selection": "first 128 images by split index (or the full split if smaller)",
                "statistics": summarize_images(validation_reference_images),
            }
            if validation_reference_images is not None
            else None
        ),
        "display_clipping": "[-1,1] applied only when saving PNGs; raw tensors are preserved",
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, allow_nan=False), encoding="utf-8")
    return metadata


def generate_cohort_from_checkpoint(
    checkpoint_path: Path,
    output_dir: Path,
    *,
    num_samples: int,
    seed: int,
    device_name: str = "auto",
    batch_size: int = 8,
    on_progress: Callable[[int, int], None] | None = None,
    validation_reference_images: torch.Tensor | None = None,
    validation_reference_selection: str = "",
) -> dict[str, Any]:
    """Generate a large cohort in bounded GPU batches and save its artifacts."""
    if not 1 <= num_samples <= 1000:
        raise ValueError("num_samples must be between 1 and 1000")
    if not 1 <= batch_size <= 8:
        raise ValueError("batch_size must be between 1 and 8")

    checkpoint_path = checkpoint_path.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = choose_device(device_name)
    checkpoint, config = load_checkpoint(checkpoint_path, device)
    diffusion = config["diffusion"]
    schedule = DiffusionSchedule(
        timesteps=int(diffusion["timesteps"]),
        beta_start=float(diffusion["beta_start"]),
        beta_end=float(diffusion["beta_end"]),
    ).to(device)
    model = build_model(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    chunks: list[torch.Tensor] = []
    chunk_seeds: list[int] = []
    for chunk_index, start in enumerate(range(0, num_samples, batch_size)):
        count = min(batch_size, num_samples - start)
        chunk_seed = (seed + chunk_index) % (2**32)
        samples, _ = sample_chain(
            model,
            schedule,
            count=count,
            seed=chunk_seed,
            device=device,
            trajectory_index=None,
            variance_type=str(diffusion.get("reverse_variance", "posterior")),
        )
        if not torch.isfinite(samples).all():
            raise RuntimeError(f"Sampling produced non-finite image values in chunk {chunk_index + 1}")
        chunks.append(samples)
        chunk_seeds.append(chunk_seed)
        if on_progress is not None:
            on_progress(min(start + count, num_samples), num_samples)

    all_samples = torch.cat(chunks, dim=0)
    torch.save(all_samples, output_dir / "samples.pt")
    display_grid(all_samples, output_dir / "samples.png", columns=min(10, num_samples))
    patch_dir = output_dir / "patches"
    patch_dir.mkdir(exist_ok=True)
    for index, sample in enumerate(all_samples):
        display_grid(sample.unsqueeze(0), patch_dir / f"patch-{index + 1:04d}.png", columns=1)
    metadata = {
        "created_at": datetime.now(UTC).isoformat(),
        "checkpoint": str(checkpoint_path),
        "checkpoint_global_step": int(checkpoint.get("global_step", -1)),
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "device": str(device),
        "seed": seed,
        "chunk_size": batch_size,
        "chunk_seeds": chunk_seeds,
        "num_samples": num_samples,
        "image_shape": [3, 64, 64],
        "timesteps": schedule.timesteps,
        "beta_start": float(diffusion["beta_start"]),
        "beta_end": float(diffusion["beta_end"]),
        "reverse_variance": str(diffusion.get("reverse_variance", "posterior")),
        "sample_value_range": [float(all_samples.min()), float(all_samples.max())],
        "fraction_outside_training_range": float((all_samples.abs() > 1.0).float().mean()),
        "sample_statistics": summarize_images(all_samples),
        "per_sample_diagnostics": [
            {
                "sample_id": f"patch-{index + 1:04d}",
                "minimum": float(sample.min()),
                "maximum": float(sample.max()),
                "mean": float(sample.mean()),
                "fraction_outside_training_range": float((sample.abs() > 1.0).float().mean()),
            }
            for index, sample in enumerate(all_samples)
        ],
        "validation_reference": (
            {
                "split": "validation",
                "selection": validation_reference_selection or "fixed unlabeled validation reference set",
                "statistics": summarize_images(validation_reference_images),
            }
            if validation_reference_images is not None
            else None
        ),
        "display_clipping": "[-1,1] applied only when saving PNGs; raw tensors are preserved",
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, allow_nan=False), encoding="utf-8"
    )
    return metadata


def run(args: argparse.Namespace) -> Path:
    checkpoint_path = args.checkpoint.resolve()
    output_dir = (args.output_dir or checkpoint_path.parent.parent / "generation" / f"seed-{args.seed}").resolve()
    validation_reference = None
    if args.compare_validation:
        from ddpm_eurosat.data import UnconditionalEuroSATSplit, load_eurosat_split

        _, config = load_checkpoint(checkpoint_path, torch.device("cpu"))
        dataset_config = config["dataset"]
        split = UnconditionalEuroSATSplit(
            load_eurosat_split("validation", dataset_config["cache_dir"])
        )
        validation_reference = torch.stack([split[index] for index in range(min(128, len(split)))])
    generate_from_checkpoint(
        checkpoint_path,
        output_dir,
        num_samples=args.num_samples,
        seed=args.seed,
        trajectory_index=args.trajectory_index,
        device_name=args.device,
        progress=True,
        validation_reference_images=validation_reference,
    )
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
    parser.add_argument(
        "--compare-validation",
        action="store_true",
        help="Add summary statistics for the first 128 images of the validation split",
    )
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
