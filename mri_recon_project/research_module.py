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
            base_channels=int(self.config["base_channels"]),
        )

    def train_batch(self, batch: dict[str, Any]) -> dict[str, Any]:
        pred, pred_normalized, target_normalized = self._predict_with_training_target(batch)
        loss = reconstruction_loss(pred_normalized, target_normalized)
        return {"loss": loss, "pred_image": pred, "logs": {"l1": loss.detach()}}

    def validate_batch(self, batch: dict[str, Any]) -> dict[str, Any]:
        pred, pred_normalized, target_normalized = self._predict_with_training_target(batch)
        loss = reconstruction_loss(pred_normalized, target_normalized)
        return {"loss": loss, "pred_image": pred, "logs": {"l1": loss.detach()}}

    def configure_optimizers(self):
        return build_optimizer(self.parameters(), float(self.config["learning_rate"]))

    def _predict(self, batch: dict[str, Any]) -> torch.Tensor:
        # Baseline receives the undersampled magnitude image and predicts a clean magnitude image.
        image = batch["zero_filled_image"].to(next(self.parameters()).device)
        normalized, mean, std = _normalize_image(image)
        pred_normalized = self.model(normalized)
        return torch.clamp(pred_normalized * std + mean, min=0.0)

    def _predict_with_training_target(self, batch: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        image = batch["zero_filled_image"].to(next(self.parameters()).device)
        target = batch["target_image"].to(image.device)
        normalized, mean, std = _normalize_image(image)
        pred_normalized = self.model(normalized)
        target_normalized = (target - mean) / std
        pred = torch.clamp(pred_normalized * std + mean, min=0.0)
        return pred, pred_normalized, target_normalized


def _normalize_image(image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mean = image.mean(dim=(-2, -1), keepdim=True)
    std = image.std(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
    return (image - mean) / std, mean, std


def build_research_module(config: dict[str, Any] | None = None) -> ResearchModule:
    return ResearchModule(config)
