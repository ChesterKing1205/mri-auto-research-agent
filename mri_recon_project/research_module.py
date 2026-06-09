from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from mri_recon_project.config import merge_project_config
from mri_recon_project.losses import reconstruction_loss
from mri_recon_project.models import SmallUNet
from mri_recon_project.optim import build_optimizer


class ResearchModule(nn.Module):
    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__()
        self.config = merge_project_config(config)
        self.model = SmallUNet(
            in_channels=int(self.config["in_channels"]),
            out_channels=int(self.config["out_channels"]),
            base_channels=int(self.config["base_channels"]),
        )

    def train_batch(self, batch: dict[str, Any]) -> dict[str, Any]:
        pred, target = self._predict_with_training_target(batch)
        loss = reconstruction_loss(_complex_magnitude(pred), target)
        return {"loss": loss, "pred_image": pred, "logs": {"l1": loss.detach()}}

    def validate_batch(self, batch: dict[str, Any]) -> dict[str, Any]:
        pred, target = self._predict_with_training_target(batch)
        loss = reconstruction_loss(_complex_magnitude(pred), target)
        return {"loss": loss, "pred_image": pred, "logs": {"l1": loss.detach()}}

    def configure_optimizers(self):
        return build_optimizer(self.parameters(), float(self.config["learning_rate"]))

    def _predict(self, batch: dict[str, Any]) -> torch.Tensor:
        image = batch["zero_filled_complex"].to(next(self.parameters()).device)
        channels = _complex_to_channels(image)
        normalized, mean, std = _normalize_image(channels)
        pred_normalized = self.model(normalized)
        pred_channels = pred_normalized * std + mean
        return _channels_to_complex(pred_channels)

    def _predict_with_training_target(self, batch: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
        pred = self._predict(batch)
        target = batch["target_image"].to(pred.device)
        return pred, target


def _normalize_image(image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mean = image.mean(dim=(-2, -1), keepdim=True)
    std = image.std(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
    return (image - mean) / std, mean, std


def _complex_to_channels(image: torch.Tensor) -> torch.Tensor:
    if image.ndim != 5 or image.shape[1] != 1 or image.shape[-1] != 2:
        raise ValueError("Expected complex image with shape (B,1,H,W,2)")
    return image[:, 0].permute(0, 3, 1, 2).contiguous()


def _channels_to_complex(channels: torch.Tensor) -> torch.Tensor:
    if channels.ndim != 4 or channels.shape[1] != 2:
        raise ValueError("Expected channel tensor with shape (B,2,H,W)")
    return channels.permute(0, 2, 3, 1).unsqueeze(1).contiguous()


def _complex_magnitude(image: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(image.square().sum(dim=-1).clamp_min(0))


def build_research_module(config: dict[str, Any] | None = None) -> ResearchModule:
    return ResearchModule(config)
