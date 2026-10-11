"""Verify the forward DDPM equation and visualize increasing noise levels."""

from __future__ import annotations

import json
import tomllib
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import torch

from ddpm_eurosat.data import UnconditionalEuroSATSplit, load_eurosat_rgb
from ddpm_eurosat.diffusion import (
    DiffusionSchedule,
    q_sample,
    q_sample_markov,
    sample_timesteps,
)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    config = tomllib.loads((root / "configs" / "base.toml").read_text(encoding="utf-8"))
    diffusion_config = config["diffusion"]
    image_size = int(config["image_size"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    schedule = DiffusionSchedule(
        timesteps=int(diffusion_config["timesteps"]),
        beta_start=float(diffusion_config["beta_start"]),
        beta_end=float(diffusion_config["beta_end"]),
    ).to(device)

    # Focused mathematical checks for this stage; no broad test framework needed.
    assert torch.isfinite(schedule.betas).all()
    assert torch.all((schedule.alphas > 0) & (schedule.alphas < 1))
    assert torch.all(schedule.alpha_bars[1:] < schedule.alpha_bars[:-1])
    assert schedule.betas.ndim == schedule.alpha_bars.ndim == 1

    # Distinct timesteps and a known noise tensor check indexing and the equation.
    check_x0 = torch.full((3, 3, image_size, image_size), 0.25, device=device)
    check_noise = torch.ones_like(check_x0)
    check_t = torch.tensor([0, schedule.timesteps // 2, schedule.timesteps - 1], device=device)
    check_xt, returned_noise = q_sample(schedule, check_x0, check_t, check_noise)
    expected = (
        schedule.extract(schedule.sqrt_alpha_bars, check_t) * check_x0
        + schedule.extract(schedule.sqrt_one_minus_alpha_bars, check_t) * check_noise
    )
    assert check_xt.shape == check_x0.shape
    assert torch.equal(returned_noise, check_noise)
    assert torch.allclose(check_xt, expected, atol=1e-7, rtol=1e-6)
    assert torch.isfinite(check_xt).all()

    random_xt, random_noise = q_sample(schedule, check_x0, check_t)
    random_expected = (
        schedule.extract(schedule.sqrt_alpha_bars, check_t) * check_x0
        + schedule.extract(schedule.sqrt_one_minus_alpha_bars, check_t) * random_noise
    )
    assert random_noise.shape == check_x0.shape
    assert torch.allclose(random_xt, random_expected, atol=1e-7, rtol=1e-6)

    timestep_generator = torch.Generator(device=device).manual_seed(int(config["seed"]))
    sampled_timesteps = sample_timesteps(
        batch_size=64,
        total_timesteps=schedule.timesteps,
        device=device,
        generator=timestep_generator,
    )
    assert sampled_timesteps.shape == (64,)
    assert sampled_timesteps.dtype == torch.long
    assert int(sampled_timesteps.min()) >= 0
    assert int(sampled_timesteps.max()) < schedule.timesteps

    # Verify a reproducible Markov chain and its first two transitions.
    markov_schedule = DiffusionSchedule(timesteps=4, beta_start=0.001, beta_end=0.01).to(device)
    markov_x0 = torch.full((1, 3, 8, 8), 0.25, device=device)
    markov_seed = 123
    trajectory = q_sample_markov(markov_schedule, markov_x0, markov_seed)
    repeated_trajectory = q_sample_markov(markov_schedule, markov_x0, markov_seed)
    markov_generator = torch.Generator(device=device).manual_seed(markov_seed)
    markov_noise = [
        torch.randn(markov_x0.shape, generator=markov_generator, device=device, dtype=markov_x0.dtype)
        for _ in range(2)
    ]
    expected_first = (
        markov_schedule.sqrt_alphas[0] * markov_x0
        + markov_schedule.sqrt_betas[0] * markov_noise[0]
    )
    expected_second = (
        markov_schedule.sqrt_alphas[1] * expected_first
        + markov_schedule.sqrt_betas[1] * markov_noise[1]
    )
    assert trajectory.shape == (markov_schedule.timesteps, *markov_x0.shape)
    assert torch.equal(trajectory, repeated_trajectory)
    assert torch.allclose(trajectory[0], expected_first, atol=1e-7, rtol=1e-6)
    assert torch.allclose(trajectory[1], expected_second, atol=1e-7, rtol=1e-6)

    # Reproducible marginal examples from one validation image with shared noise.
    source = load_eurosat_rgb(root / config["dataset"]["cache_dir"])
    validation = UnconditionalEuroSATSplit(source[config["dataset"]["validation_split"]])
    x0 = validation[0].unsqueeze(0).to(device)
    generator = torch.Generator(device=device).manual_seed(int(config["seed"]))
    noise = torch.randn(x0.shape, generator=generator, device=device, dtype=x0.dtype)
    indices = [0, 99, 249, 499, 749, schedule.timesteps - 1]

    figure, axes = plt.subplots(1, len(indices) + 1, figsize=(15, 2.8))
    original = x0[0].add(1).div(2).clamp(0, 1).permute(1, 2, 0).cpu().numpy()
    axes[0].imshow(original)
    axes[0].set_title("Original")
    for axis in axes:
        axis.axis("off")

    alpha_bars: dict[str, float] = {}
    for axis, timestep in zip(axes[1:], indices, strict=True):
        t = torch.tensor([timestep], dtype=torch.long, device=device)
        xt, _ = q_sample(schedule, x0, t, noise)
        display = xt[0].add(1).div(2).clamp(0, 1).permute(1, 2, 0).cpu().numpy()
        alpha_bar = float(schedule.alpha_bars[timestep].item())
        alpha_bars[str(timestep)] = alpha_bar
        axis.imshow(display)
        axis.set_title(f"t={timestep + 1}\nᾱ={alpha_bar:.3f}")

    figure.suptitle("Direct forward noising (same image and fixed noise)")
    figure.tight_layout()
    output_dir = root / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_path = output_dir / "eurosat_forward_noising.png"
    figure.savefig(figure_path, dpi=150, bbox_inches="tight")
    plt.close(figure)

    report = {
        "schedule": {
            "timesteps": schedule.timesteps,
            "beta_start": float(schedule.betas[0].item()),
            "beta_end": float(schedule.betas[-1].item()),
            "alpha_bar_first": float(schedule.alpha_bars[0].item()),
            "alpha_bar_last": float(schedule.alpha_bars[-1].item()),
            "finite": bool(torch.isfinite(schedule.betas).all()),
            "alpha_bounds_valid": bool(torch.all((schedule.alphas > 0) & (schedule.alphas < 1))),
            "alpha_bar_strictly_decreasing": bool(torch.all(schedule.alpha_bars[1:] < schedule.alpha_bars[:-1])),
        },
        "known_noise_equation_check": {
            "batch_shape": list(check_xt.shape),
            "timesteps_zero_based": check_t.cpu().tolist(),
            "max_absolute_error": float((check_xt - expected).abs().max().item()),
            "returned_noise_is_supplied_noise": bool(torch.equal(returned_noise, check_noise)),
        },
        "random_noise_and_timestep_checks": {
            "random_noise_shape": list(random_noise.shape),
            "random_noise_equation_max_absolute_error": float((random_xt - random_expected).abs().max().item()),
            "sampled_timestep_batch_size": int(sampled_timesteps.shape[0]),
            "sampled_timestep_min": int(sampled_timesteps.min().item()),
            "sampled_timestep_max": int(sampled_timesteps.max().item()),
            "sampled_timestep_unique_count": int(torch.unique(sampled_timesteps).numel()),
        },
        "markov_path_check": {
            "trajectory_shape": list(trajectory.shape),
            "seed": markov_seed,
            "repeat_is_identical": bool(torch.equal(trajectory, repeated_trajectory)),
            "first_two_transition_max_absolute_error": float(
                max(
                    (trajectory[0] - expected_first).abs().max().item(),
                    (trajectory[1] - expected_second).abs().max().item(),
                )
            ),
        },
        "visualization": {
            "image_source": config["dataset"]["validation_split"],
            "seed": int(config["seed"]),
            "timesteps_zero_based": indices,
            "alpha_bars": alpha_bars,
            "fixed_noise_reused": True,
            "path": str(figure_path.relative_to(root)),
        },
        "device": str(device),
    }
    report_path = output_dir / "eurosat_forward_noising_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Saved figure: {figure_path}")
    print(f"Saved report: {report_path}")


if __name__ == "__main__":
    main()
