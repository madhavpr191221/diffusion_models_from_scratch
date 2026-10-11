"""Timestep-conditioned noise prediction model for 64x64 RGB images."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class SinusoidalTimeEmbedding(nn.Module):
    """Map integer diffusion steps to cosine-then-sine features.

    For an even embedding dimension D, the first D/2 features are cosine
    terms and the remaining D/2 features are sine terms. Frequencies follow
    the usual geometric schedule from 1 down toward 1 / max_period.
    """

    def __init__(self, dimension: int = 128, max_period: float = 10_000.0) -> None:
        super().__init__()
        if dimension < 2 or dimension % 2 != 0:
            raise ValueError("dimension must be an even integer of at least 2")
        if max_period <= 1.0:
            raise ValueError("max_period must be greater than 1")
        self.dimension = dimension
        self.max_period = max_period

    def forward(self, t: Tensor) -> Tensor:
        if t.ndim != 1:
            raise ValueError(f"t must have shape [B], received {tuple(t.shape)}")
        if t.dtype != torch.long:
            raise TypeError(f"t must have dtype torch.long, received {t.dtype}")

        half_dim = self.dimension // 2
        positions = t.to(dtype=torch.float32).unsqueeze(1)
        exponents = torch.arange(half_dim, device=t.device, dtype=torch.float32)
        frequencies = torch.exp(-math.log(self.max_period) * exponents / half_dim)
        angles = positions * frequencies.unsqueeze(0)
        return torch.cat((torch.cos(angles), torch.sin(angles)), dim=1)


class TimeConditionedResidualBlock(nn.Module):
    """Two-convolution residual block with an additive time embedding."""

    def __init__(self, in_channels: int, out_channels: int, time_dimension: int = 128) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(num_groups=8, num_channels=in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.time_projection = nn.Linear(time_dimension, out_channels)
        self.norm2 = nn.GroupNorm(num_groups=8, num_channels=out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.skip_projection: nn.Module = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, kernel_size=1)
        )

    def forward(self, x: Tensor, time_embedding: Tensor) -> Tensor:
        hidden = self.conv1(F.silu(self.norm1(x)))
        hidden = hidden + self.time_projection(time_embedding)[:, :, None, None]
        hidden = self.conv2(F.silu(self.norm2(hidden)))
        return F.silu(hidden + self.skip_projection(x))


class TimeConditionedUNet(nn.Module):
    """Small unconditional U-Net that predicts the noise in ``x_t``.

    The public input contract is ``x_t: [B, 3, 64, 64]`` and ``t: [B]``
    with integer, zero-based diffusion indices. The model is conditioned on
    timestep only; no dataset labels are accepted or used.
    """

    image_shape = (3, 64, 64)
    time_dimension = 128

    def __init__(self) -> None:
        super().__init__()
        self.time_embedding = SinusoidalTimeEmbedding(self.time_dimension)
        self.time_mlp = nn.Sequential(
            nn.Linear(self.time_dimension, 512),
            nn.SiLU(),
            nn.Linear(512, self.time_dimension),
        )

        self.input_conv = nn.Conv2d(3, 32, kernel_size=3, padding=1)

        # Encoder: 64x64 -> 32x32 -> 16x16 -> 8x8.
        self.encoder_64 = TimeConditionedResidualBlock(32, 32, self.time_dimension)
        self.downsample_64_to_32 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.encoder_32 = TimeConditionedResidualBlock(64, 64, self.time_dimension)
        self.downsample_32_to_16 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.encoder_16 = TimeConditionedResidualBlock(128, 128, self.time_dimension)
        self.downsample_16_to_8 = nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1)

        self.bottleneck1 = TimeConditionedResidualBlock(128, 128, self.time_dimension)
        self.bottleneck2 = TimeConditionedResidualBlock(128, 128, self.time_dimension)

        # Decoder upsamples first, then joins the matching encoder feature map.
        self.upsample_8_to_16 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.decoder_16 = TimeConditionedResidualBlock(256, 128, self.time_dimension)
        self.upsample_16_to_32 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.decoder_32 = TimeConditionedResidualBlock(128, 64, self.time_dimension)
        self.upsample_32_to_64 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.decoder_64 = TimeConditionedResidualBlock(64, 32, self.time_dimension)

        self.output_norm = nn.GroupNorm(num_groups=8, num_channels=32)
        self.output_conv = nn.Conv2d(32, 3, kernel_size=3, padding=1)

    def forward(self, x_t: Tensor, t: Tensor) -> Tensor:
        if x_t.ndim != 4 or tuple(x_t.shape[1:]) != self.image_shape:
            raise ValueError(
                f"x_t must have shape [B, 3, 64, 64], received {tuple(x_t.shape)}"
            )
        if not x_t.is_floating_point():
            raise TypeError(f"x_t must be floating point, received {x_t.dtype}")
        if t.shape != (x_t.shape[0],):
            raise ValueError(f"t must have shape [{x_t.shape[0]}], received {tuple(t.shape)}")
        if t.dtype != torch.long:
            raise TypeError(f"t must have dtype torch.long, received {t.dtype}")
        if t.device != x_t.device:
            raise ValueError("x_t and t must be on the same device")
        if x_t.device != self.input_conv.weight.device:
            raise ValueError("Move the model and x_t to the same device before calling it")

        time_embedding = self.time_mlp(self.time_embedding(t))

        skip_64 = self.encoder_64(self.input_conv(x_t), time_embedding)
        skip_32 = self.encoder_32(self.downsample_64_to_32(skip_64), time_embedding)
        skip_16 = self.encoder_16(self.downsample_32_to_16(skip_32), time_embedding)
        hidden = self.downsample_16_to_8(skip_16)
        hidden = self.bottleneck1(hidden, time_embedding)
        hidden = self.bottleneck2(hidden, time_embedding)

        hidden = F.interpolate(hidden, scale_factor=2, mode="nearest")
        hidden = self.upsample_8_to_16(hidden)
        hidden = self.decoder_16(torch.cat((hidden, skip_16), dim=1), time_embedding)

        hidden = F.interpolate(hidden, scale_factor=2, mode="nearest")
        hidden = self.upsample_16_to_32(hidden)
        hidden = self.decoder_32(torch.cat((hidden, skip_32), dim=1), time_embedding)

        hidden = F.interpolate(hidden, scale_factor=2, mode="nearest")
        hidden = self.upsample_32_to_64(hidden)
        hidden = self.decoder_64(torch.cat((hidden, skip_64), dim=1), time_embedding)

        return self.output_conv(F.silu(self.output_norm(hidden)))
