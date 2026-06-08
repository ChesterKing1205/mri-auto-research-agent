from __future__ import annotations

import contextlib
from pathlib import Path
from typing import Any

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from mri_recon_harness.config import ProgramConfig
from mri_recon_harness.datamodule import FastMRIDataModule
from mri_recon_harness.lightning_module import HarnessLightningModule
from mri_recon_harness.manifest import write_manifests


def run_experiment(config: ProgramConfig) -> dict[str, Any]:
    pl.seed_everything(config.seed, workers=True)

    if not Path("manifests/train_manifest.tsv").exists() or not Path("manifests/val_manifest.tsv").exists():
        write_manifests(config)

    datamodule = FastMRIDataModule(config)
    module = HarnessLightningModule()
    logger = TensorBoardLogger("outputs", name="")
    output_dir = Path(logger.log_dir)
    checkpoint_dir = output_dir / "checkpoint"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    callbacks = [
        ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename="epoch{epoch}-psnr{val_metrics/psnr:.4f}",
            monitor="val_metrics/psnr",
            mode="max",
            save_top_k=3,
            auto_insert_metric_name=False,
        ),
        ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename="epoch{epoch}-ssim{val_metrics/ssim:.4f}",
            monitor="val_metrics/ssim",
            mode="max",
            save_top_k=3,
            auto_insert_metric_name=False,
        ),
        ModelCheckpoint(dirpath=checkpoint_dir, filename="last", save_last=True, save_top_k=0),
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
    log_path = output_dir / "log.txt"
    with log_path.open("w", encoding="utf-8") as log_file:
        with contextlib.redirect_stdout(log_file), contextlib.redirect_stderr(log_file):
            trainer.fit(module, datamodule=datamodule)

    metrics = trainer.callback_metrics
    return {
        "psnr": float(metrics["val_metrics/psnr"].detach().cpu()),
        "ssim": float(metrics["val_metrics/ssim"].detach().cpu()),
        "nmse": float(metrics["val_metrics/nmse"].detach().cpu()),
        "val_loss": float(metrics["loss/val_loss"].detach().cpu()),
        "output_dir": str(output_dir),
    }
