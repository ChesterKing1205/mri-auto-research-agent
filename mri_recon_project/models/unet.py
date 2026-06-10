from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        num_layers: int = 2,
        activation: str = "leaky_relu",
        normalization: str = "instance",
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")

        layers: list[nn.Module] = []
        for layer_idx in range(num_layers):
            conv_in = in_channels if layer_idx == 0 else out_channels
            layers.append(nn.Conv2d(conv_in, out_channels, kernel_size=3, padding=1))
            norm = _make_normalization(normalization, out_channels)
            if norm is not None:
                layers.append(norm)
            layers.append(_make_activation(activation))
        self.net = nn.Sequential(*layers)
        self.skip = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x) + self.skip(x)


class Downsample(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.op = nn.Conv2d(channels, channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(x)


class Upsample(nn.Module):
    def __init__(self, channels: int, mode: str) -> None:
        super().__init__()
        if mode == "transpose":
            self.op = nn.ConvTranspose2d(channels, channels, kernel_size=4, stride=2, padding=1)
        elif mode == "bilinear":
            self.op = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            )
        else:
            raise ValueError(f"Unsupported upsample_mode: {mode}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(x)


class SmallUNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 2,
        base_channels: int = 16,
        *,
        depth: int = 3,
        channel_multiplier: int = 2,
        conv_layers_per_block: int = 2,
        activation: str = "leaky_relu",
        normalization: str = "instance",
        upsample_mode: str = "transpose",
    ) -> None:
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be >= 1")
        if base_channels < 1:
            raise ValueError("base_channels must be positive")
        if channel_multiplier < 1:
            raise ValueError("channel_multiplier must be positive")

        channels = [base_channels * (channel_multiplier**level) for level in range(depth)]
        self.in_conv = nn.Conv2d(in_channels, channels[0], kernel_size=3, padding=1)

        self.down_blocks = nn.ModuleList()
        self.downs = nn.ModuleList()
        prev_channels = channels[0]
        for idx, channels_at_level in enumerate(channels):
            self.down_blocks.append(
                ResidualBlock(
                    prev_channels,
                    channels_at_level,
                    num_layers=conv_layers_per_block,
                    activation=activation,
                    normalization=normalization,
                )
            )
            is_last_level = idx == len(channels) - 1
            self.downs.append(nn.Identity() if is_last_level else Downsample(channels_at_level))
            prev_channels = channels_at_level

        self.mid = ResidualBlock(
            channels[-1],
            channels[-1],
            num_layers=conv_layers_per_block,
            activation=activation,
            normalization=normalization,
        )

        self.ups = nn.ModuleList()
        self.up_blocks = nn.ModuleList()
        reversed_channels = list(reversed(channels))
        prev_channels = reversed_channels[0]
        for idx, skip_channels in enumerate(reversed_channels):
            self.ups.append(nn.Identity() if idx == 0 else Upsample(prev_channels, upsample_mode))
            self.up_blocks.append(
                ResidualBlock(
                    prev_channels + skip_channels,
                    skip_channels,
                    num_layers=conv_layers_per_block,
                    activation=activation,
                    normalization=normalization,
                )
            )
            prev_channels = skip_channels

        self.out = nn.Sequential(
            _make_normalization(normalization, channels[0]) or nn.Identity(),
            _make_activation(activation),
            nn.Conv2d(channels[0], out_channels, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.in_conv(x)
        skips = []
        for block, down in zip(self.down_blocks, self.downs, strict=True):
            x = block(x)
            skips.append(x)
            x = down(x)

        x = self.mid(x)

        for up, block in zip(self.ups, self.up_blocks, strict=True):
            x = up(x)
            skip = skips.pop()
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode="nearest")
            x = block(torch.cat([x, skip], dim=1))
        return self.out(x)


def _make_activation(name: str) -> nn.Module:
    if name == "silu":
        return nn.SiLU()
    if name == "leaky_relu":
        return nn.LeakyReLU(negative_slope=0.1, inplace=True)
    if name == "relu":
        return nn.ReLU(inplace=True)
    if name == "gelu":
        return nn.GELU()
    raise ValueError(f"Unsupported activation: {name}")


def _make_normalization(name: str, channels: int) -> nn.Module | None:
    if name == "group":
        return nn.GroupNorm(_group_count(channels), channels)
    if name == "instance":
        return nn.InstanceNorm2d(channels, affine=True)
    if name == "batch":
        return nn.BatchNorm2d(channels)
    if name == "none":
        return None
    raise ValueError(f"Unsupported normalization: {name}")


def _group_count(channels: int) -> int:
    for groups in (8, 4, 2):
        if channels % groups == 0:
            return groups
    return 1
