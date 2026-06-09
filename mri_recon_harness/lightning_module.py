from __future__ import annotations

from typing import Any

import pytorch_lightning as pl
import torch

from mri_recon_harness.metrics import compute_metrics
from mri_recon_harness.physics import complex_abs
from mri_recon_project import build_research_module


class HarnessLightningModule(pl.LightningModule):
    def __init__(self, project_config: dict[str, Any] | None = None) -> None:
        super().__init__()
        self.project = build_research_module(project_config or {})
        self.save_hyperparameters({"project_config": project_config or {}})

    def training_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        output = self.project.train_batch(batch)
        loss = _require_tensor(output, "loss")
        pred = _require_tensor(output, "pred_image")
        pred_magnitude = _prediction_magnitude(pred)
        target = batch["target_image"].to(pred_magnitude.device)
        if pred_magnitude.shape != target.shape:
            raise ValueError(
                f"pred_image magnitude shape {tuple(pred_magnitude.shape)} does not match target {tuple(target.shape)}"
            )
        self.log("loss/train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        _log_extra(self, output.get("logs", {}), prefix="loss/train")
        return loss

    def validation_step(self, batch: dict[str, Any], batch_idx: int) -> dict[str, torch.Tensor]:
        output = self.project.validate_batch(batch)
        loss = _require_tensor(output, "loss")
        pred = _require_tensor(output, "pred_image")
        pred_magnitude = _prediction_magnitude(pred)
        target = batch["target_image"].to(pred_magnitude.device)
        if pred_magnitude.shape != target.shape:
            raise ValueError(
                f"pred_image magnitude shape {tuple(pred_magnitude.shape)} does not match target {tuple(target.shape)}"
            )
        metrics = compute_metrics(pred_magnitude, target)
        self.log("loss/val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_metrics/psnr", metrics["psnr"], on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_metrics/ssim", metrics["ssim"], on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_metrics/nmse", metrics["nmse"], on_step=False, on_epoch=True, prog_bar=True)
        _log_extra(self, output.get("logs", {}), prefix="project/val")
        return {"val_loss": loss, **metrics}

    def configure_optimizers(self):
        return self.project.configure_optimizers()


def _require_tensor(output: dict[str, Any], key: str) -> torch.Tensor:
    value = output.get(key)
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"Project output must include tensor field '{key}'")
    return value


def _prediction_magnitude(pred: torch.Tensor) -> torch.Tensor:
    if pred.ndim == 5 and pred.shape[1] == 1 and pred.shape[-1] == 2:
        return complex_abs(pred)
    raise ValueError("pred_image must have shape (B,1,H,W,2) for complex output")


def _log_extra(module: pl.LightningModule, logs: dict[str, Any], *, prefix: str) -> None:
    for key, value in logs.items():
        if isinstance(value, torch.Tensor) and value.ndim == 0:
            module.log(f"{prefix}_{key}", value, on_step=False, on_epoch=True, prog_bar=False)
