from __future__ import annotations

import contextlib
from pathlib import Path
from typing import Any

import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from mri_recon_harness.config import ProgramConfig
from mri_recon_harness.datamodule import FastMRIDataModule
from mri_recon_harness.lightning_module import HarnessLightningModule
from mri_recon_harness.manifest import write_manifests
from mri_recon_project.config import merge_project_config


class ValidationMetricLog(Callback):
    def __init__(self, log_path: Path) -> None:
        super().__init__()
        self.log_path = log_path

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if trainer.sanity_checking:
            return
        metrics = trainer.callback_metrics
        required = ["val_metrics/psnr", "val_metrics/ssim", "val_metrics/nmse", "loss/val_loss"]
        if not all(key in metrics for key in required):
            return
        line = (
            f"epoch={trainer.current_epoch} "
            f"psnr={_metric_value(metrics['val_metrics/psnr']):.6f} "
            f"ssim={_metric_value(metrics['val_metrics/ssim']):.6f} "
            f"nmse={_metric_value(metrics['val_metrics/nmse']):.6f} "
            f"val_loss={_metric_value(metrics['loss/val_loss']):.6f}\n"
        )
        with self.log_path.open("a", encoding="utf-8") as log_file:
            log_file.write(line)


def run_experiment(config: ProgramConfig) -> dict[str, Any]:
    project_config = merge_project_config(None)
    seed = int(project_config["seed"])
    pl.seed_everything(seed, workers=True)

    if not Path("manifests/train_manifest.tsv").exists() or not Path("manifests/val_manifest.tsv").exists():
        write_manifests(config)

    datamodule = FastMRIDataModule(config, seed)
    module = HarnessLightningModule(project_config)
    logger = TensorBoardLogger("outputs", name="")
    output_dir = Path(logger.log_dir)
    checkpoint_dir = output_dir / "checkpoint"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "log.txt"
    log_path.write_text("", encoding="utf-8")
    callbacks = [
        ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename="epoch{epoch}-psnr{val_metrics/psnr:.4f}",
            monitor="val_metrics/psnr",
            mode="max",
            save_top_k=0,
            auto_insert_metric_name=False,
        ),
        ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename="epoch{epoch}-ssim{val_metrics/ssim:.4f}",
            monitor="val_metrics/ssim",
            mode="max",
            save_top_k=0,
            auto_insert_metric_name=False,
        ),
        ModelCheckpoint(dirpath=checkpoint_dir, filename="last", save_last=True, save_top_k=0),
        ValidationMetricLog(log_path),
    ]
    trainer = pl.Trainer(
        max_epochs=config.epochs_per_round,
        accelerator="auto",
        devices=1,
        logger=logger,
        callbacks=callbacks,
        enable_progress_bar=False,
        log_every_n_steps=1,
        deterministic=True,
    )
    with (output_dir / "trainer.log").open("w", encoding="utf-8") as trainer_log:
        with contextlib.redirect_stdout(trainer_log), contextlib.redirect_stderr(trainer_log):
            trainer.fit(module, datamodule=datamodule)

    metrics = trainer.callback_metrics
    return {
        "psnr": float(metrics["val_metrics/psnr"].detach().cpu()),
        "ssim": float(metrics["val_metrics/ssim"].detach().cpu()),
        "nmse": float(metrics["val_metrics/nmse"].detach().cpu()),
        "val_loss": float(metrics["loss/val_loss"].detach().cpu()),
        "output_dir": str(output_dir),
    }


def _metric_value(value: Any) -> float:
    return float(value.detach().cpu())
