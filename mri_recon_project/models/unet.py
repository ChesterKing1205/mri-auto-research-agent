from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DownBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        conv_layers: int,
        activation: str,
        normalization: str,
    ) -> None:
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.block = ConvBlock(
            in_channels,
            out_channels,
            num_layers=conv_layers,
            activation=activation,
            normalization=normalization,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(self.pool(x))


class UpBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        *,
        conv_layers: int,
        activation: str,
        normalization: str,
        upsample_mode: str,
    ) -> None:
        super().__init__()
        self.up = _make_upsample(upsample_mode, in_channels, out_channels)
        self.block = ConvBlock(
            out_channels + skip_channels,
            out_channels,
            num_layers=conv_layers,
            activation=activation,
            normalization=normalization,
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = _match_size(self.up(x), skip)
        return self.block(torch.cat([x, skip], dim=1))


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

        channels = [base_channels * (channel_multiplier**level) for level in range(depth + 1)]
        block_kwargs = {
            "conv_layers": conv_layers_per_block,
            "activation": activation,
            "normalization": normalization,
        }
        self.encoder = nn.ModuleList(
            [
                ConvBlock(
                    in_channels,
                    channels[0],
                    num_layers=conv_layers_per_block,
                    activation=activation,
                    normalization=normalization,
                ),
                *[
                    DownBlock(channels[level - 1], channels[level], **block_kwargs)
                    for level in range(1, depth)
                ],
            ]
        )
        self.bottleneck = DownBlock(channels[depth - 1], channels[depth], **block_kwargs)
        self.decoder = nn.ModuleList(
            [
                UpBlock(
                    channels[level + 1],
                    channels[level],
                    channels[level],
                    **block_kwargs,
                    upsample_mode=upsample_mode,
                )
                for level in range(depth - 1, -1, -1)
            ]
        )
        self.out = nn.Conv2d(channels[0], out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        for block in self.encoder:
            x = block(x)
            skips.append(x)

        x = self.bottleneck(x)
        for block, skip in zip(self.decoder, reversed(skips), strict=True):
            x = block(x, skip)
        return self.out(x)


def _make_activation(name: str) -> nn.Module:
    if name == "leaky_relu":
        return nn.LeakyReLU(negative_slope=0.1, inplace=True)
    if name == "relu":
        return nn.ReLU(inplace=True)
    if name == "silu":
        return nn.SiLU(inplace=True)
    if name == "gelu":
        return nn.GELU()
    raise ValueError(f"Unsupported activation: {name}")


def _make_normalization(name: str, channels: int) -> nn.Module | None:
    if name == "instance":
        return nn.InstanceNorm2d(channels, affine=True)
    if name == "batch":
        return nn.BatchNorm2d(channels)
    if name == "none":
        return None
    raise ValueError(f"Unsupported normalization: {name}")


def _make_upsample(name: str, in_channels: int, out_channels: int) -> nn.Module:
    if name == "transpose":
        return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
    if name == "bilinear":
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
        )
    raise ValueError(f"Unsupported upsample_mode: {name}")


def _match_size(x: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    if x.shape[-2:] == reference.shape[-2:]:
        return x
    return F.interpolate(x, size=reference.shape[-2:], mode="bilinear", align_corners=False)
