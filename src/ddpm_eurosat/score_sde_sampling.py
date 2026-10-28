"""Unconditional generation with the continuous VP-SDE reverse process."""

from __future__ import annotations

import argparse
import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable

import torch
from torch import Tensor, nn

from ddpm_eurosat.model import build_model
from ddpm_eurosat.sampling import LIVE_FRAME_STRIDE, display_grid, summarize_images
from ddpm_eurosat.vp_sde import VPSDESchedule, epsilon_to_score

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def predict_score(
    model: nn.Module, x_t: Tensor, t: Tensor, schedule: VPSDESchedule, prediction: str,
) -> Tensor:
    """Return a score from either model parameterization, explicitly."""
    prediction_value = model(x_t, t)
    if prediction == "score":
        return prediction_value
    if prediction == "epsilon":
        return epsilon_to_score(schedule, prediction_value, t)
    raise ValueError("prediction must be 'epsilon' or 'score'")


@torch.inference_mode()
def sample_reverse_sde(
    model: nn.Module,
    schedule: VPSDESchedule,
    *,
    count: int,
    seed: int,
    device: torch.device,
    sampling_steps: int = 1000,
    prediction: str = "epsilon",
    trajectory_index: int | None = 0,
    on_frame: Callable[[int, int, float, Tensor], None] | None = None,
    progress: bool = False,
) -> tuple[Tensor, Tensor | None]:
    r"""Sample using Euler--Maruyama for the reverse-time VP-SDE.

    With ``f(x,t)=-0.5*beta(t)*x`` and ``g(t)^2=beta(t)``, integrate backward
    over ``dt=-h`` as

    ``x_next = x - (f(x,t)-beta(t)*score(x,t))*h + sqrt(beta(t)*h)*z``.

    The score is evaluated only at positive upper endpoints. Brownian noise
    is included on every interval, including the final interval ending at 0.
    A device-local generator controls the initial Gaussian and all increments.
    """
    if count < 1:
        raise ValueError("count must be at least 1")
    if sampling_steps < 1:
        raise ValueError("sampling_steps must be at least 1")
    if trajectory_index is not None and not 0 <= trajectory_index < count:
        raise ValueError("trajectory_index must be None or in [0, count)")

    generator = torch.Generator(device=device).manual_seed(seed)
    current = torch.randn((count, 3, 64, 64), generator=generator, device=device)
    trajectory = [current[trajectory_index].cpu()] if trajectory_index is not None else None
    if on_frame is not None and trajectory_index is not None:
        on_frame(0, 0, 1.0, current[trajectory_index].cpu())

    was_training = model.training
    model.eval()
    step_size = 1.0 / sampling_steps
    started = time.perf_counter()
    try:
        for step in range(sampling_steps):
            # t_i = 1 - i/N. The last query is at t=1/N, never t=0.
            t_value = 1.0 - step / sampling_steps
            t = torch.full((count,), t_value, dtype=torch.float32, device=device)
            beta = schedule.beta(t).reshape(count, 1, 1, 1)
            score = predict_score(model, current, t, schedule, prediction)
            drift = -0.5 * beta * current
            brownian = torch.randn(current.shape, generator=generator, device=device, dtype=current.dtype)

            # Reverse integration over a positive interval of length h.
            current = current - (drift - beta * score) * step_size
            current = current + torch.sqrt(beta * step_size) * brownian

            step_completed = step + 1
            frame_t = max(1.0 - step_completed / sampling_steps, 0.0)
            if trajectory is not None:
                trajectory.append(current[trajectory_index].cpu())
            if (
                on_frame is not None
                and trajectory_index is not None
                and (step_completed % LIVE_FRAME_STRIDE == 0 or step_completed == sampling_steps)
            ):
                on_frame(step_completed // LIVE_FRAME_STRIDE, step_completed, frame_t, current[trajectory_index].cpu())
            if progress and (step_completed % 100 == 0 or step_completed == sampling_steps):
                print(f"VP-SDE reverse steps: {step_completed}/{sampling_steps}", flush=True)
    finally:
        model.train(was_training)

    elapsed = time.perf_counter() - started
    print(f"VP-SDE sampling completed in {elapsed:.1f}s ({count} samples, {sampling_steps} steps)", flush=True)
    return current.cpu(), torch.stack(trajectory) if trajectory is not None else None


@torch.inference_mode()
def sample_annealed_langevin(
    model: nn.Module,
    schedule: VPSDESchedule,
    *,
    count: int,
    seed: int,
    device: torch.device,
    prediction: str = "epsilon",
    levels: int = 100,
    steps_per_level: int = 10,
    training_time_min: float = 1e-5,
    base_step_size: float = 2e-5,
    trajectory_index: int | None = 0,
    on_frame: Callable[[int, int, float, Tensor], None] | None = None,
) -> tuple[Tensor, Tensor | None]:
    r"""Annealed Langevin dynamics over VP noise levels.

    At each descending noise level, repeat
    ``x <- x + (eta/2) score_theta(x,t) + sqrt(eta) z`` with
    ``eta(t) = base_step_size * sigma(t)^2 / sigma(1)^2``. This is a
    standalone corrector-style sampler, not a reverse-SDE predictor-corrector.
    """
    if count < 1 or levels < 2 or steps_per_level < 1:
        raise ValueError("count >= 1, levels >= 2, and steps_per_level >= 1 are required")
    if not 0.0 < training_time_min < 1.0 or base_step_size <= 0:
        raise ValueError("training_time_min and base_step_size must be positive (t_min < 1)")
    total_steps = levels * steps_per_level
    if trajectory_index is not None and not 0 <= trajectory_index < count:
        raise ValueError("trajectory_index must be None or in [0, count)")

    generator = torch.Generator(device=device).manual_seed(seed)
    current = torch.randn((count, 3, 64, 64), generator=generator, device=device)
    trajectory = [current[trajectory_index].cpu()] if trajectory_index is not None else None
    if on_frame is not None and trajectory_index is not None:
        on_frame(0, 0, 1.0, current[trajectory_index].cpu())

    times = torch.linspace(1.0, training_time_min, levels, device=device)
    sigma_one = schedule.marginal_std(torch.ones(1, device=device))[0]
    was_training = model.training
    model.eval()
    step = 0
    try:
        for level_index, t_value in enumerate(times):
            t = torch.full((count,), float(t_value), dtype=torch.float32, device=device)
            sigma = schedule.marginal_std(t)[0]
            eta = base_step_size * (sigma.square() / sigma_one.square())
            for _ in range(steps_per_level):
                score = predict_score(model, current, t, schedule, prediction)
                noise = torch.randn(current.shape, generator=generator, device=device, dtype=current.dtype)
                current = current + 0.5 * eta * score + torch.sqrt(eta) * noise
                step += 1
                if trajectory is not None:
                    trajectory.append(current[trajectory_index].cpu())
                if on_frame is not None and trajectory_index is not None and step % LIVE_FRAME_STRIDE == 0:
                    on_frame(step // LIVE_FRAME_STRIDE, step, float(t_value), current[trajectory_index].cpu())
    finally:
        model.train(was_training)
    return current.cpu(), torch.stack(trajectory) if trajectory is not None else None


def generate_from_checkpoint(
    checkpoint_path: Path,
    output_dir: Path,
    *,
    num_samples: int,
    seed: int,
    device_name: str = "auto",
    on_frame: Callable[[int, int, float, Tensor], None] | None = None,
    validation_reference_images: Tensor | None = None,
    validation_reference_selection: str | None = None,
    sampling_steps: int | None = None,
    sampler: str = "reverse_sde",
    langevin_levels: int = 100,
    langevin_steps_per_level: int = 10,
) -> dict[str, Any]:
    """Load a VP-SDE checkpoint, sample, and persist standard run artifacts."""
    from ddpm_eurosat.sampling import choose_device

    checkpoint_path = checkpoint_path.resolve()
    device = choose_device(device_name)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint or "config" not in checkpoint:
        raise ValueError("Checkpoint does not contain the expected model state and resolved config")
    config = checkpoint["config"]
    if config.get("process") != "vp_sde" or config.get("model", {}).get("architecture") not in {
        "vp_sde_ddpm_cifar_v1", "vp_sde_direct_score_ddpm_cifar_v1",
    }:
        raise ValueError("Checkpoint is not a compatible VP-SDE score model")
    sde_config = config["vp_sde"]
    steps = int(sampling_steps or sde_config.get("sampling_steps", 1000))
    schedule = VPSDESchedule(float(sde_config["beta_min"]), float(sde_config["beta_max"]))
    model = build_model(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    output_dir.mkdir(parents=True, exist_ok=True)

    prediction = str(config["model"].get("prediction", "epsilon"))
    if prediction not in {"epsilon", "score"}:
        raise ValueError("VP-SDE checkpoint prediction must be epsilon or score")
    started = time.perf_counter()
    if sampler == "reverse_sde":
        samples, trajectory = sample_reverse_sde(
            model, schedule, count=num_samples, seed=seed, device=device,
            sampling_steps=steps, prediction=prediction, trajectory_index=0, on_frame=on_frame,
        )
        network_evaluations = steps
    elif sampler == "annealed_langevin":
        levels = int(langevin_levels)
        inner_steps = int(langevin_steps_per_level)
        samples, trajectory = sample_annealed_langevin(
            model, schedule, count=num_samples, seed=seed, device=device,
            prediction=prediction, levels=levels, steps_per_level=inner_steps,
            training_time_min=float(sde_config["training_time_min"]),
            trajectory_index=0, on_frame=on_frame,
        )
        network_evaluations = levels * inner_steps
        steps = network_evaluations
    else:
        raise ValueError("sampler must be 'reverse_sde' or 'annealed_langevin'")
    elapsed = time.perf_counter() - started
    if not torch.isfinite(samples).all():
        raise RuntimeError("VP-SDE sampling produced non-finite image values")
    torch.save(samples, output_dir / "samples.pt")
    display_grid(samples, output_dir / "samples.png", columns=min(4, num_samples))
    assert trajectory is not None
    torch.save(trajectory, output_dir / "trajectory.pt")
    positions = torch.linspace(0, steps, steps=min(9, steps + 1)).round().long()
    display_grid(trajectory[positions], output_dir / "trajectory.png", columns=3)

    metadata: dict[str, Any] = {
        "created_at": datetime.now(UTC).isoformat(),
        "process": "vp_sde",
        "objective": "sigma_squared_weighted_dsm_epsilon_mse" if prediction == "epsilon" else "sigma_squared_weighted_dsm_direct_score",
        "prediction_parameterization": prediction,
        "sampler": sampler,
        "checkpoint": str(checkpoint_path),
        "checkpoint_global_step": int(checkpoint.get("global_step", -1)),
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "device": str(device),
        "seed": seed,
        "num_samples": num_samples,
        "image_shape": [3, 64, 64],
        "beta_min": schedule.beta_min,
        "beta_max": schedule.beta_max,
        "sampling_steps": steps,
        "network_evaluations": network_evaluations,
        "time_grid": "t_i = 1 - i/N; score evaluated at i=0..N-1" if sampler == "reverse_sde" else f"{levels} descending VP noise levels from 1 to t_min; {inner_steps} Langevin updates per level",
        "terminal_prior": "standard Gaussian N(0,I), approximating the VP marginal at t=1",
        "phi_at_t1": float(schedule.marginal_mean_coefficient(torch.ones(1))[0]),
        "trajectory_index": 0,
        "trajectory_frame_order": "x_1 to x_0" if sampler == "reverse_sde" else "initial Gaussian x_1 through descending VP noise levels to final Langevin state",
        "trajectory_frames": int(trajectory.shape[0]),
        "elapsed_seconds": elapsed,
        "sample_value_range": [float(samples.min()), float(samples.max())],
        "fraction_outside_training_range": float((samples.abs() > 1.0).float().mean()),
        "sample_statistics": summarize_images(samples),
        "per_sample_diagnostics": [
            {
                "sample_id": f"patch-{index + 1:04d}",
                "minimum": float(sample.min()),
                "maximum": float(sample.max()),
                "mean": float(sample.mean()),
                "fraction_outside_training_range": float((sample.abs() > 1.0).float().mean()),
            }
            for index, sample in enumerate(samples)
        ],
    }
    if sampler == "annealed_langevin":
        metadata.update({
            "langevin_levels": levels,
            "langevin_steps_per_level": inner_steps,
            "langevin_base_step_size": 2e-5,
            "langevin_step_size": "eta(t)=2e-5*sigma(t)^2/sigma(1)^2",
            "training_time_min": float(sde_config["training_time_min"]),
        })
    if validation_reference_images is not None:
        metadata["validation_reference"] = {
            "split": "validation",
            "selection": validation_reference_selection or "first validation images by split index",
            "statistics": summarize_images(validation_reference_images),
        }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, allow_nan=False), encoding="utf-8")
    return metadata


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--trajectory-index", type=int, default=0)
    parser.add_argument("--sampling-steps", type=int)
    parser.add_argument("--sampler", choices=("reverse_sde", "annealed_langevin"), default="reverse_sde")
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.num_samples < 1:
        raise SystemExit("--num-samples must be at least 1")
    if args.trajectory_index != 0:
        raise SystemExit("The saved trajectory currently follows sample 0")
    checkpoint_path = args.checkpoint.resolve()
    output_dir = (args.output_dir or checkpoint_path.parent.parent / "generation" / f"seed-{args.seed}").resolve()
    generate_from_checkpoint(
        checkpoint_path, output_dir, num_samples=args.num_samples, seed=args.seed,
        device_name=args.device, sampling_steps=args.sampling_steps,
        sampler=args.sampler,
    )


if __name__ == "__main__":
    main()
