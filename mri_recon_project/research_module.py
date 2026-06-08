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
        pred = self._predict(batch)
        target = batch["target_image"].to(pred.device)
        loss = reconstruction_loss(pred, target)
        return {"loss": loss, "pred_image": pred, "logs": {"l1": loss.detach()}}

    def validate_batch(self, batch: dict[str, Any]) -> dict[str, Any]:
        pred = self._predict(batch)
        target = batch["target_image"].to(pred.device)
        loss = reconstruction_loss(pred, target)
        return {"loss": loss, "pred_image": pred, "logs": {"l1": loss.detach()}}

    def configure_optimizers(self):
        return build_optimizer(self.parameters(), float(self.config["learning_rate"]))

    def _predict(self, batch: dict[str, Any]) -> torch.Tensor:
        # Baseline starts from normalized zero-filled magnitude image supplied by the fixed harness.
        image = batch["zero_filled_image"].to(next(self.parameters()).device)
        return self.model(image)


def build_research_module(config: dict[str, Any] | None = None) -> ResearchModule:
    return ResearchModule(config)
