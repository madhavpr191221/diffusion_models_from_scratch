"""Continuous-time variance-preserving SDE utilities.

This module is intentionally separate from :mod:`diffusion`, whose schedule
contains discrete DDPM beta values indexed by integer timesteps.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional as F


@dataclass(frozen=True)
class VPSDESchedule:
    r"""Linear VP-SDE rate with closed-form marginal coefficients.

    For ``beta(t) = beta_min + t * (beta_max - beta_min)``, the integrated
    rate is ``beta_min*t + 0.5*(beta_max-beta_min)*t**2``. Therefore
    ``Phi(t) = exp(-0.5 * integrated_beta(t))`` and
    ``sigma(t) = sqrt(1 - Phi(t)**2)``.

    The scalar endpoints are configuration values, not discrete beta tables.
    Time tensors accepted by methods have shape ``[B]`` and live on the same
    device as the image/model tensors that consume their outputs.
    """

    beta_min: float = 0.1
    beta_max: float = 20.0

    def __post_init__(self) -> None:
        if not 0.0 < self.beta_min <= self.beta_max:
            raise ValueError("Require 0 < beta_min <= beta_max")

    @staticmethod
    def _check_time(t: Tensor) -> None:
        if t.ndim != 1:
            raise ValueError(f"t must have shape [B], received {tuple(t.shape)}")
        if not t.is_floating_point():
            raise TypeError(f"t must be floating point, received {t.dtype}")

    def beta(self, t: Tensor) -> Tensor:
        """Evaluate the continuous rate ``beta(t)`` for each batch item."""
        self._check_time(t)
        return self.beta_min + t * (self.beta_max - self.beta_min)

    def integrated_beta(self, t: Tensor) -> Tensor:
        """Return ``integral_0^t beta(s) ds`` in closed form."""
        self._check_time(t)
        return self.beta_min * t + 0.5 * (self.beta_max - self.beta_min) * t.square()

    def marginal_mean_coefficient(self, t: Tensor) -> Tensor:
        """Return ``Phi(t) = exp(-0.5 * integral_0^t beta(s) ds)``."""
        return torch.exp(-0.5 * self.integrated_beta(t))

    def marginal_std(self, t: Tensor) -> Tensor:
        """Return the exact VP marginal standard deviation ``sqrt(1-Phi^2)``."""
        # -expm1 is more accurate than 1 - exp near t=0.
        return torch.sqrt(-torch.expm1(-self.integrated_beta(t)))


def sample_continuous_times(
    batch_size: int,
    *,
    device: torch.device | str,
    t_min: float = 1e-5,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Draw independent uniform continuous times in ``[t_min, 1)``.

    The explicit generator lets callers reproduce time draws without relying
    on PyTorch's global random state. ``torch.rand`` supplies values in
    ``[0, 1)``; scaling them avoids the singular score conversion at ``t=0``.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    if not 0.0 < t_min < 1.0:
        raise ValueError("t_min must be in (0, 1)")
    unit_times = torch.rand(
        (batch_size,), device=device, dtype=torch.float32, generator=generator
    )
    return t_min + (1.0 - t_min) * unit_times


def _expand_batch(values: Tensor, image: Tensor) -> Tensor:
    if image.ndim != 4:
        raise ValueError(f"image must have shape [B,C,H,W], received {tuple(image.shape)}")
    if values.shape != (image.shape[0],):
        raise ValueError(f"coefficient must have shape [{image.shape[0]}], received {tuple(values.shape)}")
    if values.device != image.device:
        raise ValueError("time coefficients and image must be on the same device")
    return values.reshape(image.shape[0], 1, 1, 1)


def q_sample_vp(
    schedule: VPSDESchedule,
    x0: Tensor,
    t: Tensor,
    noise: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    r"""Sample the exact forward marginal ``q(x_t | x_0)``.

    Implements ``x_t = Phi(t) * x0 + sigma(t) * epsilon``. Supplied noise is
    returned unchanged, making the corruption directly inspectable and easy
    to reproduce in equation checks.
    """
    if x0.ndim != 4 or not x0.is_floating_point():
        raise ValueError("x0 must be a floating [B,C,H,W] tensor")
    if t.shape != (x0.shape[0],):
        raise ValueError(f"t must have shape [{x0.shape[0]}], received {tuple(t.shape)}")
    if t.device != x0.device:
        raise ValueError("x0 and t must be on the same device")
    if noise is None:
        noise = torch.randn_like(x0)
    elif noise.shape != x0.shape or noise.device != x0.device or noise.dtype != x0.dtype:
        raise ValueError("noise must match x0 shape, device, and dtype")

    phi = _expand_batch(schedule.marginal_mean_coefficient(t), x0)
    sigma = _expand_batch(schedule.marginal_std(t), x0)
    return phi * x0 + sigma * noise, noise


def epsilon_to_score(
    schedule: VPSDESchedule,
    epsilon_prediction: Tensor,
    t: Tensor,
) -> Tensor:
    r"""Convert predicted noise to the score: ``s_theta = -eps_theta/sigma``.

    The score is singular at ``t=0``. Training samples times from ``[t_min,1)``
    and reverse-SDE code must likewise not request a score at exactly zero.
    """
    sigma = schedule.marginal_std(t)
    expanded_sigma = _expand_batch(sigma, epsilon_prediction)
    if not epsilon_prediction.is_floating_point():
        raise TypeError("epsilon_prediction must be floating point")
    if t.shape != (epsilon_prediction.shape[0],):
        raise ValueError(
            f"t must have shape [{epsilon_prediction.shape[0]}], received {tuple(t.shape)}"
        )
    return -epsilon_prediction / expanded_sigma


def conditional_score_target(schedule: VPSDESchedule, noise: Tensor, t: Tensor) -> Tensor:
    r"""Return the conditional DSM target ``-epsilon / sigma(t)``."""
    sigma = _expand_batch(schedule.marginal_std(t), noise)
    return -noise / sigma


def weighted_dsm_loss(
    schedule: VPSDESchedule,
    score_prediction: Tensor,
    noise: Tensor,
    t: Tensor,
) -> Tensor:
    r"""Compute sigma-squared weighted conditional denoising score matching.

    The target is ``-epsilon/sigma``. Since
    ``s_theta = -epsilon_theta/sigma``, the weighted elementwise residual is
    ``sigma**2 * (s_theta + epsilon/sigma)**2 =
    (epsilon_theta - epsilon)**2``. Thus this scalar equals ordinary epsilon
    MSE for predictions converted through :func:`epsilon_to_score`.
    """
    if score_prediction.shape != noise.shape:
        raise ValueError("score_prediction and noise must have identical shapes")
    if score_prediction.device != noise.device or score_prediction.dtype != noise.dtype:
        raise ValueError("score_prediction and noise must match device and dtype")
    sigma = _expand_batch(schedule.marginal_std(t), noise)
    target = -noise / sigma
    return (sigma.square() * (score_prediction - target).square()).mean()


def epsilon_prediction_loss(epsilon_prediction: Tensor, noise: Tensor) -> Tensor:
    """Return the epsilon-MSE equal to sigma-squared weighted conditional DSM."""
    if epsilon_prediction.shape != noise.shape:
        raise ValueError("epsilon_prediction and noise must have identical shapes")
    return F.mse_loss(epsilon_prediction, noise)
