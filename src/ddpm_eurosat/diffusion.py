"""Forward-process coefficients and direct DDPM noising."""

from __future__ import annotations

import torch
from torch import Tensor
from torch import nn


class DiffusionSchedule(nn.Module):
    """Linear DDPM schedule with zero-based tensor indexing for timesteps 1..T."""

    def __init__(
        self,
        timesteps: int = 1_000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
    ) -> None:
        super().__init__()
        if timesteps < 1:
            raise ValueError("timesteps must be at least 1")
        if not 0.0 < beta_start <= beta_end < 1.0:
            raise ValueError("Require 0 < beta_start <= beta_end < 1")

        betas = torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float32)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)

        # Buffers move with the schedule to CPU/GPU but are not learned parameters.
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bars", alpha_bars)
        self.register_buffer("sqrt_betas", torch.sqrt(betas))
        self.register_buffer("sqrt_alphas", torch.sqrt(alphas))
        self.register_buffer("sqrt_alpha_bars", torch.sqrt(alpha_bars))
        self.register_buffer("sqrt_one_minus_alpha_bars", torch.sqrt(1.0 - alpha_bars))
        self.register_buffer("sqrt_recip_alphas", torch.rsqrt(alphas))

    @property
    def timesteps(self) -> int:
        return int(self.betas.numel())

    def extract(self, coefficients: Tensor, t: Tensor, image_ndim: int = 4) -> Tensor:
        """Select one scalar per batch item and shape it for image broadcasting."""
        if t.ndim != 1:
            raise ValueError(f"t must have shape [B], received {tuple(t.shape)}")
        if t.dtype != torch.long:
            raise TypeError(f"t must have dtype torch.long, received {t.dtype}")
        if image_ndim < 2:
            raise ValueError("image_ndim must be at least 2")
        if t.device != coefficients.device:
            raise ValueError("t and schedule coefficients must be on the same device")
        if t.numel() and (int(t.min()) < 0 or int(t.max()) >= coefficients.numel()):
            raise IndexError(f"timestep indices must be in [0, {coefficients.numel() - 1}]")

        selected = torch.index_select(coefficients, 0, t)
        return selected.reshape(t.shape[0], *([1] * (image_ndim - 1)))


def sample_timesteps(
    batch_size: int,
    total_timesteps: int,
    *,
    device: torch.device | str,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Draw an independent uniform zero-based timestep for each batch item."""
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    if total_timesteps < 1:
        raise ValueError("total_timesteps must be at least 1")
    return torch.randint(
        low=0,
        high=total_timesteps,
        size=(batch_size,),
        device=device,
        generator=generator,
        dtype=torch.long,
    )


def q_sample(
    schedule: DiffusionSchedule,
    x0: Tensor,
    t: Tensor,
    noise: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Draw x_t directly from q(x_t|x_0); return x_t and its exact noise."""
    if x0.ndim != 4:
        raise ValueError(f"x0 must have shape [B,C,H,W], received {tuple(x0.shape)}")
    if not x0.is_floating_point():
        raise TypeError(f"x0 must be floating point, received {x0.dtype}")
    if t.shape != (x0.shape[0],):
        raise ValueError(f"t must have shape [{x0.shape[0]}], received {tuple(t.shape)}")
    if x0.device != schedule.betas.device or t.device != schedule.betas.device:
        raise ValueError("Move the schedule, x0, and t to the same device before q_sample")

    if noise is None:
        noise = torch.randn_like(x0)
    elif noise.shape != x0.shape:
        raise ValueError(f"noise shape {tuple(noise.shape)} must match x0 {tuple(x0.shape)}")
    elif noise.device != x0.device or noise.dtype != x0.dtype:
        raise ValueError("noise must match x0 device and dtype")

    signal = schedule.extract(schedule.sqrt_alpha_bars, t, x0.ndim)
    noise_scale = schedule.extract(schedule.sqrt_one_minus_alpha_bars, t, x0.ndim)
    return signal * x0 + noise_scale * noise, noise


@torch.no_grad()
def q_sample_markov(
    schedule: DiffusionSchedule,
    x0: Tensor,
    seed: int,
) -> Tensor:
    """Generate one seeded Markov forward path, returning states x_1 through x_T.

    The returned tensor has shape [T,B,C,H,W]. Each transition uses fresh
    Gaussian noise from one generator, making the complete path reproducible.
    """
    if x0.ndim != 4:
        raise ValueError(f"x0 must have shape [B,C,H,W], received {tuple(x0.shape)}")
    if not x0.is_floating_point():
        raise TypeError(f"x0 must be floating point, received {x0.dtype}")
    if x0.device != schedule.betas.device:
        raise ValueError("Move the schedule and x0 to the same device before sampling")

    generator = torch.Generator(device=x0.device).manual_seed(seed)
    current = x0
    states: list[Tensor] = []
    for step in range(schedule.timesteps):
        noise = torch.randn(
            x0.shape,
            generator=generator,
            device=x0.device,
            dtype=x0.dtype,
        )
        current = schedule.sqrt_alphas[step] * current + schedule.sqrt_betas[step] * noise
        states.append(current)
    return torch.stack(states, dim=0)
