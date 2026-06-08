from __future__ import annotations

import h5py
import numpy as np

from mri_recon_harness.config import load_program_config
from mri_recon_harness.manifest import read_manifest, write_manifests


def _write_fastmri_file(path, slices: int = 3) -> None:
    kspace = (np.random.randn(slices, 2, 640, 320) + 1j * np.random.randn(slices, 2, 640, 320)).astype(
        np.complex64
    )
    target = np.abs(np.random.randn(slices, 320, 320)).astype(np.float32)
    with h5py.File(path, "w") as hf:
        hf.create_dataset("kspace", data=kspace)
        hf.create_dataset("reconstruction_rss", data=target)


def test_prepare_manifest_writes_file_slice_rows(tmp_path, monkeypatch):
    root = tmp_path / "fastmri"
    train = root / "multicoil_train"
    val = root / "multicoil_val"
    train.mkdir(parents=True)
    val.mkdir(parents=True)
    _write_fastmri_file(train / "train.h5")
    _write_fastmri_file(val / "val.h5")
    program = tmp_path / "program.md"
    program.write_text(
        f"""
fastMRI_root: {root}
train_files: 1
val_files: 1
epochs_per_round: 1
batch_size: 1
max_minutes_per_round: 15
acceleration: 4
center_fraction: 0.08
seed: 123
""",
        encoding="utf-8",
    )
    config = load_program_config(program)
    train_manifest, val_manifest = write_manifests(config, tmp_path / "manifests")
    assert len(read_manifest(train_manifest)) == 3
    assert len(read_manifest(val_manifest)) == 3
