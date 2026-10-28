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


class DDPMResidualBlock(nn.Module):
    """Residual block matching Ho et al.'s CIFAR-10 DDPM U-Net."""

    def __init__(self, in_channels: int, out_channels: int, time_dimension: int, dropout: float) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.time_projection = nn.Linear(time_dimension, out_channels)
        self.norm2 = nn.GroupNorm(num_groups=32, num_channels=out_channels, eps=1e-6)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)
        self.skip = nn.Identity() if in_channels == out_channels else nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x: Tensor, time_embedding: Tensor) -> Tensor:
        hidden = self.conv1(F.silu(self.norm1(x)))
        hidden = hidden + self.time_projection(F.silu(time_embedding))[:, :, None, None]
        hidden = self.conv2(self.dropout(F.silu(self.norm2(hidden))))
        return self.skip(x) + hidden


class DDPMAttentionBlock(nn.Module):
    """Full spatial self-attention block used by the original DDPM U-Net."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(num_groups=32, num_channels=channels, eps=1e-6)
        self.query = nn.Conv2d(channels, channels, kernel_size=1)
        self.key = nn.Conv2d(channels, channels, kernel_size=1)
        self.value = nn.Conv2d(channels, channels, kernel_size=1)
        self.projection = nn.Conv2d(channels, channels, kernel_size=1)
        nn.init.zeros_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)

    def forward(self, x: Tensor) -> Tensor:
        batch, channels, height, width = x.shape
        normalized = self.norm(x)
        query = self.query(normalized).reshape(batch, channels, height * width)
        key = self.key(normalized).reshape(batch, channels, height * width)
        value = self.value(normalized).reshape(batch, channels, height * width)
        weights = torch.einsum("bct,bcs->bts", query * (channels ** -0.5), key).softmax(dim=-1)
        attended = torch.einsum("bts,bcs->bct", weights, value).reshape(batch, channels, height, width)
        return x + self.projection(attended)


class OriginalDDPMUNet(nn.Module):
    """Unconditional CIFAR-10 DDPM U-Net, adapted from 32x32 to 64x64 inputs.

    The four resolution levels are 64, 32, 16, and 8 pixels. Attention is used
    at 16x16 and in the bottleneck, as in the authors' released implementation.
    """

    image_shape = (3, 64, 64)
    base_channels = 128
    channel_multipliers = (1, 2, 2, 2)
    time_dimension = 512

    def __init__(self, dropout: float = 0.1) -> None:
        super().__init__()
        ch = self.base_channels
        self.time_mlp = nn.Sequential(
            nn.Linear(ch, ch * 4),
            nn.Linear(ch * 4, ch * 4),
        )
        self.input_conv = nn.Conv2d(3, ch, 3, padding=1)

        levels = []
        input_channels = ch
        for level, multiplier in enumerate(self.channel_multipliers):
            output_channels = ch * multiplier
            blocks = nn.ModuleList()
            attentions = nn.ModuleList()
            for _ in range(2):
                blocks.append(DDPMResidualBlock(input_channels, output_channels, self.time_dimension, dropout))
                input_channels = output_channels
                attentions.append(DDPMAttentionBlock(output_channels) if level == 2 else nn.Identity())
            downsample = (
                nn.Conv2d(input_channels, input_channels, 3, stride=2, padding=1)
                if level < len(self.channel_multipliers) - 1
                else nn.Identity()
            )
            levels.append(nn.ModuleDict({"blocks": blocks, "attentions": attentions, "downsample": downsample}))
        self.down_levels = nn.ModuleList(levels)

        self.middle1 = DDPMResidualBlock(input_channels, input_channels, self.time_dimension, dropout)
        self.middle_attention = DDPMAttentionBlock(input_channels)
        self.middle2 = DDPMResidualBlock(input_channels, input_channels, self.time_dimension, dropout)

        up_levels = []
        for level in reversed(range(len(self.channel_multipliers))):
            output_channels = ch * self.channel_multipliers[level]
            blocks = nn.ModuleList()
            attentions = nn.ModuleList()
            for block_index in range(3):
                skip_channels = (
                    ch
                    if (level == 1 and block_index == 2) or level == 0
                    else ch * self.channel_multipliers[level]
                )
                blocks.append(DDPMResidualBlock(input_channels + skip_channels, output_channels, self.time_dimension, dropout))
                input_channels = output_channels
                attentions.append(DDPMAttentionBlock(output_channels) if level == 2 else nn.Identity())
            upsample = (
                nn.Conv2d(input_channels, input_channels, 3, padding=1)
                if level > 0
                else nn.Identity()
            )
            up_levels.append(nn.ModuleDict({"blocks": blocks, "attentions": attentions, "upsample": upsample}))
        self.up_levels = nn.ModuleList(up_levels)
        self.output_norm = nn.GroupNorm(num_groups=32, num_channels=ch, eps=1e-6)
        self.output_conv = nn.Conv2d(ch, 3, 3, padding=1)
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        for module in self.modules():
            if isinstance(module, DDPMResidualBlock):
                nn.init.zeros_(module.conv2.weight)
                nn.init.zeros_(module.conv2.bias)
            elif isinstance(module, DDPMAttentionBlock):
                nn.init.zeros_(module.projection.weight)
                nn.init.zeros_(module.projection.bias)
        nn.init.zeros_(self.output_conv.weight)
        nn.init.zeros_(self.output_conv.bias)

    def forward(self, x_t: Tensor, t: Tensor) -> Tensor:
        if x_t.ndim != 4 or tuple(x_t.shape[1:]) != self.image_shape:
            raise ValueError(f"x_t must have shape [B,3,64,64], received {tuple(x_t.shape)}")
        if not x_t.is_floating_point() or t.shape != (x_t.shape[0],) or t.dtype != torch.long:
            raise ValueError("x_t must be floating point and t must be a [B] long tensor")
        if x_t.device != t.device or x_t.device != self.input_conv.weight.device:
            raise ValueError("x_t, t, and model must be on the same device")

        time_embedding = self._embed_time_positions(t.to(torch.float32))
        return self._forward_with_time_embedding(x_t, time_embedding)

    def _embed_time_positions(self, positions: Tensor) -> Tensor:
        """Embed numeric time positions with the original DDPM Fourier features."""
        half_dim = self.base_channels // 2
        frequency_exponents = torch.arange(half_dim, device=positions.device, dtype=torch.float32)
        frequencies = torch.exp(-math.log(10_000.0) * frequency_exponents / (half_dim - 1))
        angles = positions.unsqueeze(1) * frequencies.unsqueeze(0)
        time_embedding = torch.cat((torch.sin(angles), torch.cos(angles)), dim=1)
        time_embedding = self.time_mlp[0](time_embedding)
        return self.time_mlp[1](F.silu(time_embedding))

    def _forward_with_time_embedding(self, x_t: Tensor, time_embedding: Tensor) -> Tensor:
        hidden = self.input_conv(x_t)
        skips = [hidden]
        for level in self.down_levels:
            for block, attention in zip(level["blocks"], level["attentions"]):
                hidden = attention(block(hidden, time_embedding))
                skips.append(hidden)
            hidden = level["downsample"](hidden)
            if not isinstance(level["downsample"], nn.Identity):
                skips.append(hidden)

        hidden = self.middle2(self.middle_attention(self.middle1(hidden, time_embedding)), time_embedding)
        for level_index, level in enumerate(self.up_levels):
            original_level = len(self.up_levels) - 1 - level_index
            for block, attention in zip(level["blocks"], level["attentions"]):
                skip = skips.pop()
                if hidden.shape[-2:] != skip.shape[-2:]:
                    hidden = F.interpolate(hidden, size=skip.shape[-2:], mode="nearest")
                hidden = attention(block(torch.cat((hidden, skip), dim=1), time_embedding))
            if original_level > 0:
                hidden = F.interpolate(hidden, scale_factor=2, mode="nearest")
                hidden = level["upsample"](hidden)

        if skips:
            raise RuntimeError(f"U-Net left {len(skips)} unmatched skip tensors")
        return self.output_conv(F.silu(self.output_norm(hidden)))


class ContinuousTimeOriginalDDPMUNet(OriginalDDPMUNet):
    """Original DDPM U-Net parameterized by continuous VP-SDE time.

    This subclass adds no learned parameters. It maps ``t in [0,1]`` to the
    original sinusoidal embedding's ``[0,999]`` phase range, then uses exactly
    the same U-Net backbone and time MLP. Its output is predicted epsilon.
    """

    def forward(self, x_t: Tensor, t: Tensor) -> Tensor:
        if x_t.ndim != 4 or tuple(x_t.shape[1:]) != self.image_shape:
            raise ValueError(f"x_t must have shape [B,3,64,64], received {tuple(x_t.shape)}")
        if not x_t.is_floating_point():
            raise TypeError(f"x_t must be floating point, received {x_t.dtype}")
        if t.shape != (x_t.shape[0],):
            raise ValueError(f"t must have shape [{x_t.shape[0]}], received {tuple(t.shape)}")
        if not t.is_floating_point():
            raise TypeError(f"t must be floating point, received {t.dtype}")
        if x_t.device != t.device or x_t.device != self.input_conv.weight.device:
            raise ValueError("x_t, t, and model must be on the same device")

        # Continuous normalized SDE time shares the original model's 0..999
        # Fourier phase range without changing discrete integer-time behavior.
        positions = t.to(torch.float32) * 999.0
        time_embedding = self._embed_time_positions(positions)
        return self._forward_with_time_embedding(x_t, time_embedding)


class ContinuousTimeDirectScoreUNet(ContinuousTimeOriginalDDPMUNet):
    """Same continuous-time U-Net, trained to emit the DSM score directly.

    The tensor shape and learned architecture match the epsilon model. The
    output meaning is recorded in the run config: this model predicts
    ``-epsilon / sigma(t)`` rather than epsilon.
    """


def build_model(config: dict | None = None) -> nn.Module:
    """Construct a model from saved configuration; absent IDs mean legacy compact U-Net."""
    model_config = (config or {}).get("model", {})
    architecture = model_config.get("architecture", "compact_v1")
    if architecture == "ddpm_cifar_v1":
        return OriginalDDPMUNet(dropout=float(model_config.get("dropout", 0.1)))
    if architecture == "vp_sde_ddpm_cifar_v1":
        return ContinuousTimeOriginalDDPMUNet(dropout=float(model_config.get("dropout", 0.1)))
    if architecture == "vp_sde_direct_score_ddpm_cifar_v1":
        return ContinuousTimeDirectScoreUNet(dropout=float(model_config.get("dropout", 0.1)))
    if architecture == "compact_v1":
        return TimeConditionedUNet()
    raise ValueError(f"Unknown model architecture: {architecture}")
