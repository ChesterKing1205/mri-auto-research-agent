from __future__ import annotations

import h5py
import numpy as np

from mri_recon_harness.config import ProgramConfig
from mri_recon_harness.datamodule import FastMRISliceDataset
from mri_recon_harness.guardrails import find_frozen_changes
from mri_recon_harness.manifest import write_manifests


def _write_fastmri_file(path, slices: int = 1) -> None:
    kspace = (np.random.randn(slices, 2, 640, 320) + 1j * np.random.randn(slices, 2, 640, 320)).astype(
        np.complex64
    )
    target = np.abs(np.random.randn(slices, 320, 320)).astype(np.float32)
    with h5py.File(path, "w") as hf:
        hf.create_dataset("kspace", data=kspace)
        hf.create_dataset("reconstruction_rss", data=target)


def test_dataset_reads_required_fields_without_cache(tmp_path):
    root = tmp_path / "fastmri"
    train = root / "multicoil_train"
    val = root / "multicoil_val"
    train.mkdir(parents=True)
    val.mkdir(parents=True)
    _write_fastmri_file(train / "train.h5")
    _write_fastmri_file(val / "val.h5")
    config = ProgramConfig(
        fastmri_root=root,
        train_files=1,
        val_files=1,
        epochs_per_round=1,
        batch_size=1,
        max_minutes_per_round=15,
        acceleration=4,
        center_fraction=0.08,
        seed=1,
    )
    train_manifest, _ = write_manifests(config, tmp_path / "manifests")
    sample = FastMRISliceDataset(train_manifest, config)[0]
    assert sample["masked_kspace"].shape[-1] == 2
    assert sample["full_kspace"].shape[-3:-1] == (320, 320)
    assert sample["target_image"].shape == (1, 320, 320)
    assert sample["normalization_info"]["scale"].item() > 0
    assert sample["fname"] == "train.h5"


def test_guardrail_flags_non_project_changes():
    frozen = find_frozen_changes(
        [
            "mri_recon_project/research_module.py",
            "results.tsv",
            "mri_recon_harness/metrics.py",
            "program.md",
        ]
    )
    assert frozen == ["mri_recon_harness/metrics.py", "program.md"]
