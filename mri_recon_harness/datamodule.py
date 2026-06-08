from __future__ import annotations

from pathlib import Path
from typing import Any

import h5py
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset

from mri_recon_harness.config import ProgramConfig
from mri_recon_harness.manifest import read_manifest
from mri_recon_harness.physics import (
    center_crop_real,
    ifft2c,
    make_equispaced_mask,
    rss,
    standardize_kspace,
    to_tensor_complex_last,
)


class FastMRISliceDataset(Dataset):
    def __init__(self, manifest_path: str | Path, config: ProgramConfig) -> None:
        self.rows = read_manifest(manifest_path)
        self.config = config

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        file_path, slice_num = self.rows[idx]
        with h5py.File(file_path, "r") as hf:
            raw_kspace = hf["kspace"][slice_num]
            target = hf["reconstruction_rss"][slice_num]

        kspace = to_tensor_complex_last(raw_kspace)
        if kspace.shape[-3:-1] != (320, 320):
            kspace = standardize_kspace(kspace.unsqueeze(0)).squeeze(0)

        full_image = ifft2c(kspace.unsqueeze(0)).squeeze(0)
        zero_filled_target = rss(full_image.unsqueeze(0), dim=1).squeeze(0)

        target_image = torch.from_numpy(target).float()
        target_image = center_crop_real(target_image, (320, 320)).unsqueeze(0)
        scale = target_image.max().clamp_min(1e-6)
        target_image = target_image / scale

        width = kspace.shape[-2]
        mask_1d = make_equispaced_mask(width, self.config.acceleration, self.config.center_fraction)
        mask = mask_1d.view(1, 1, width, 1).expand(kspace.shape[0], kspace.shape[1], width, 2)
        masked_kspace = kspace * mask

        zero_filled_target = center_crop_real(zero_filled_target, (320, 320)).unsqueeze(0) / scale

        return {
            "masked_kspace": masked_kspace,
            "full_kspace": kspace,
            "mask": mask,
            "target_image": target_image,
            "zero_filled_image": zero_filled_target,
            "normalization_info": {"scale": scale},
            "fname": file_path.name,
            "slice_num": slice_num,
        }


def _collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    tensor_keys = [
        "masked_kspace",
        "full_kspace",
        "mask",
        "target_image",
        "zero_filled_image",
    ]
    for key in tensor_keys:
        out[key] = torch.stack([item[key] for item in batch], dim=0)
    out["normalization_info"] = {"scale": torch.stack([item["normalization_info"]["scale"] for item in batch])}
    out["fname"] = [item["fname"] for item in batch]
    out["slice_num"] = torch.tensor([item["slice_num"] for item in batch], dtype=torch.long)
    return out


class FastMRIDataModule(pl.LightningDataModule):
    def __init__(self, config: ProgramConfig, manifest_dir: str | Path = "manifests") -> None:
        super().__init__()
        self.config = config
        self.manifest_dir = Path(manifest_dir)
        self.train_dataset: FastMRISliceDataset | None = None
        self.val_dataset: FastMRISliceDataset | None = None

    def setup(self, stage: str | None = None) -> None:
        self.train_dataset = FastMRISliceDataset(self.manifest_dir / "train_manifest.tsv", self.config)
        self.val_dataset = FastMRISliceDataset(self.manifest_dir / "val_manifest.tsv", self.config)

    def train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            self.setup("fit")
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            collate_fn=_collate,
        )

    def val_dataloader(self) -> DataLoader:
        if self.val_dataset is None:
            self.setup("validate")
        return DataLoader(
            self.val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            collate_fn=_collate,
        )

