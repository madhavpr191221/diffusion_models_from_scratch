"""Unconditional EuroSAT conditional flow matching in noise-to-data time."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from ddpm_eurosat.model import TimeConditionedResidualBlock
from ddpm_eurosat.ode import ODESolution, SolverMethod, solve_ode

BEST_FLOW_MATCHING_CHECKPOINT = "best-eurosat-flow-matching.pt"
BEST_FLOW_MATCHING_EMA_CHECKPOINT = "best-eurosat-flow-matching-ema.pt"
LATEST_FLOW_MATCHING_CHECKPOINT = "latest-eurosat-flow-matching.pt"


class ContinuousTimeEmbedding(nn.Module):
    """Fourier features for normalized flow time in [0, 1]."""

    def __init__(self, dimension: int = 128, max_period: float = 10_000.0) -> None:
        super().__init__()
        if dimension < 2 or dimension % 2:
            raise ValueError("dimension must be even and at least two")
        self.dimension = dimension
        self.max_period = max_period

    def forward(self, time: Tensor) -> Tensor:
        if time.ndim != 1 or not time.is_floating_point():
            raise ValueError("flow time must be a floating tensor with shape [B]")
        if not torch.isfinite(time).all() or torch.any((time < 0) | (time > 1)):
            raise ValueError("flow time values must lie in [0, 1]")
        half = self.dimension // 2
        exponent = torch.arange(half, device=time.device, dtype=torch.float32) / max(half - 1, 1)
        frequencies = torch.exp(-math.log(self.max_period) * exponent)
        phase = (time.to(torch.float32) * 999.0).unsqueeze(1) * frequencies.unsqueeze(0)
        return torch.cat((torch.sin(phase), torch.cos(phase)), dim=1)


class ContinuousTimeVelocityUNet(nn.Module):
    """Compact image U-Net whose output is dY/dt, not an image or noise.

    The continuous time clock is noise at t=0 and data at t=1. The image
    input and velocity output both have shape [B, 3, 64, 64].
    """

    image_shape = (3, 64, 64)
    time_dimension = 128
    architecture_id = "flow_matching_compact_v1"

    def __init__(self) -> None:
        super().__init__()
        self.time_embedding = ContinuousTimeEmbedding(self.time_dimension)
        self.time_mlp = nn.Sequential(
            nn.Linear(self.time_dimension, 512), nn.SiLU(), nn.Linear(512, self.time_dimension)
        )
        self.input_conv = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.encoder_64 = TimeConditionedResidualBlock(32, 32, self.time_dimension)
        self.downsample_64_to_32 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.encoder_32 = TimeConditionedResidualBlock(64, 64, self.time_dimension)
        self.downsample_32_to_16 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.encoder_16 = TimeConditionedResidualBlock(128, 128, self.time_dimension)
        self.downsample_16_to_8 = nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1)
        self.bottleneck1 = TimeConditionedResidualBlock(128, 128, self.time_dimension)
        self.bottleneck2 = TimeConditionedResidualBlock(128, 128, self.time_dimension)
        self.upsample_8_to_16 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.decoder_16 = TimeConditionedResidualBlock(256, 128, self.time_dimension)
        self.upsample_16_to_32 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.decoder_32 = TimeConditionedResidualBlock(128, 64, self.time_dimension)
        self.upsample_32_to_64 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.decoder_64 = TimeConditionedResidualBlock(64, 32, self.time_dimension)
        self.output_norm = nn.GroupNorm(num_groups=8, num_channels=32)
        self.output_conv = nn.Conv2d(32, 3, kernel_size=3, padding=1)

    def forward(self, x: Tensor, time: Tensor) -> Tensor:
        if x.ndim != 4 or tuple(x.shape[1:]) != self.image_shape or not x.is_floating_point():
            raise ValueError("x must be a floating tensor with shape [B, 3, 64, 64]")
        if time.shape != (x.shape[0],) or not time.is_floating_point():
            raise ValueError("time must be a floating tensor with shape [B]")
        if x.device != time.device or x.device != self.input_conv.weight.device:
            raise ValueError("x, time, and model must be on the same device")
        embedding = self.time_mlp(self.time_embedding(time))
        skip_64 = self.encoder_64(self.input_conv(x), embedding)
        skip_32 = self.encoder_32(self.downsample_64_to_32(skip_64), embedding)
        skip_16 = self.encoder_16(self.downsample_32_to_16(skip_32), embedding)
        hidden = self.bottleneck1(self.downsample_16_to_8(skip_16), embedding)
        hidden = self.bottleneck2(hidden, embedding)
        hidden = self.upsample_8_to_16(F.interpolate(hidden, scale_factor=2, mode="nearest"))
        hidden = self.decoder_16(torch.cat((hidden, skip_16), dim=1), embedding)
        hidden = self.upsample_16_to_32(F.interpolate(hidden, scale_factor=2, mode="nearest"))
        hidden = self.decoder_32(torch.cat((hidden, skip_32), dim=1), embedding)
        hidden = self.upsample_32_to_64(F.interpolate(hidden, scale_factor=2, mode="nearest"))
        hidden = self.decoder_64(torch.cat((hidden, skip_64), dim=1), embedding)
        return self.output_conv(F.silu(self.output_norm(hidden)))


def dequantize_normalized_pixels(
    images: Tensor,
    *,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Jitter normalized uint8 pixel centers within their width-2/255 cells."""
    if not images.is_floating_point():
        raise TypeError("images must be floating point normalized tensors")
    jitter = torch.rand(images.shape, device=images.device, dtype=images.dtype, generator=generator)
    return images + (jitter - 0.5) * (2.0 / 255.0)


def flow_matching_path(
    noise: Tensor,
    data: Tensor,
    time: Tensor,
) -> tuple[Tensor, Tensor]:
    """Return Y_t=(1-t)Z+tX and its conditional velocity target X-Z."""
    if noise.shape != data.shape or noise.ndim != 4:
        raise ValueError("noise and data must have the same image-batch shape")
    if time.shape != (noise.shape[0],) or not time.is_floating_point():
        raise ValueError("time must be a floating tensor with shape [B]")
    if time.device != noise.device or data.device != noise.device:
        raise ValueError("noise, data, and time must be on the same device")
    if torch.any((time < 0) | (time > 1)):
        raise ValueError("time values must lie in [0, 1]")
    weight = time.to(noise.dtype).view(-1, 1, 1, 1)
    return (1.0 - weight) * noise + weight * data, data - noise


def flow_matching_loss(
    model: nn.Module,
    data: Tensor,
    *,
    noise: Tensor | None = None,
    time: Tensor | None = None,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Conditional flow-matching MSE, averaged over batch and scalar pixels."""
    if noise is None:
        noise = torch.randn(data.shape, device=data.device, dtype=data.dtype, generator=generator)
    if time is None:
        time = torch.rand(data.shape[0], device=data.device, dtype=data.dtype, generator=generator)
    points, target = flow_matching_path(noise, data, time)
    prediction = model(points, time)
    if prediction.shape != target.shape:
        raise ValueError("velocity model output must have the input image shape")
    return F.mse_loss(prediction, target)


@torch.inference_mode()
def sample_flow_matching(
    model: nn.Module,
    *,
    count: int,
    seed: int,
    steps: int = 100,
    method: SolverMethod = "heun",
    device: torch.device | str = "cpu",
    save_every: int = 1,
    initial_noise: Tensor | None = None,
) -> tuple[Tensor, ODESolution]:
    """Draw seeded Gaussian images and transport them from t=0 to t=1."""
    if count < 1 or steps < 1:
        raise ValueError("count and steps must be positive")
    device = torch.device(device)
    if initial_noise is None:
        generator = torch.Generator(device=device).manual_seed(seed)
        initial_noise = torch.randn((count, 3, 64, 64), generator=generator, device=device)
    elif initial_noise.shape != (count, 3, 64, 64):
        raise ValueError("initial_noise must have shape [count, 3, 64, 64]")
    else:
        initial_noise = initial_noise.to(device)
    time_grid = torch.linspace(0.0, 1.0, steps + 1, device=device)
    model.eval()
    trajectory = solve_ode(
        lambda time, state: model(state, time.expand(state.shape[0])),
        initial_noise,
        time_grid,
        method=method,
        save_every=save_every,
    )
    return trajectory.states[-1], trajectory
