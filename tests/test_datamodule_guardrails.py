from __future__ import annotations

import h5py
import numpy as np
import torch

from mri_recon_harness.config import ProgramConfig
from mri_recon_harness.datamodule import FastMRISliceDataset
from mri_recon_harness.guardrails import find_frozen_changes
from mri_recon_harness.manifest import write_manifests
from mri_recon_harness.physics import complex_abs, estimate_sens_maps, sens_reduce


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
        acs=24,
        seed=1,
    )
    train_manifest, _ = write_manifests(config, tmp_path / "manifests")
    sample = FastMRISliceDataset(train_manifest, config)[0]
    assert sample["masked_kspace"].shape[-1] == 2
    assert sample["full_kspace"].shape[-3:-1] == (320, 320)
    assert sample["mask"].shape == (1, 1, 320, 1)
    assert sample["sens_maps"].shape == sample["full_kspace"].shape
    assert sample["target_complex"].shape == (1, 320, 320, 2)
    assert sample["zero_filled_complex"].shape == (1, 320, 320, 2)
    assert sample["target_image"].shape == (1, 320, 320)
    assert sample["zero_filled_image"].shape == (1, 320, 320)
    assert sample["normalization_info"]["scale"].item() > 0
    assert sample["fname"] == "train.h5"


def test_zero_filled_image_is_reconstructed_from_masked_kspace(tmp_path):
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
        acs=24,
        seed=1,
    )
    train_manifest, _ = write_manifests(config, tmp_path / "manifests")
    sample = FastMRISliceDataset(train_manifest, config)[0]

    masked_batch = sample["masked_kspace"].unsqueeze(0)
    mask_batch = sample["mask"].unsqueeze(0)
    sens_maps = estimate_sens_maps(masked_batch, mask_batch)
    expected_complex = sens_reduce(masked_batch, sens_maps) / sample["normalization_info"]["scale"]
    expected = complex_abs(expected_complex).squeeze(0)

    wrong_full_complex = sens_reduce(sample["full_kspace"].unsqueeze(0), sens_maps)
    wrong_full_image = complex_abs(wrong_full_complex).squeeze(0) / sample["normalization_info"]["scale"]

    assert torch.allclose(sample["zero_filled_image"], expected)
    assert torch.allclose(sample["zero_filled_complex"], expected_complex.squeeze(0))
    assert not torch.allclose(sample["zero_filled_image"], wrong_full_image)


def test_target_image_comes_from_reconstruction_rss_not_sens_reduce(tmp_path):
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
        acs=24,
        seed=1,
    )
    train_manifest, _ = write_manifests(config, tmp_path / "manifests")
    sample = FastMRISliceDataset(train_manifest, config)[0]

    sens_target = complex_abs(sample["target_complex"].unsqueeze(0)).squeeze(0)
    assert not torch.allclose(sample["target_image"], sens_target)


def test_mask_marks_sampled_kspace_positions(tmp_path):
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
        acs=24,
        seed=1,
    )
    train_manifest, _ = write_manifests(config, tmp_path / "manifests")
    sample = FastMRISliceDataset(train_manifest, config)[0]

    assert set(torch.unique(sample["mask"]).tolist()) <= {0.0, 1.0}
    assert torch.allclose(sample["masked_kspace"], sample["full_kspace"] * sample["mask"])
    assert int(sample["mask"].sum().item()) == 80
    center_start = 320 // 2 - config.acs // 2
    center_end = center_start + config.acs
    assert torch.all(sample["mask"][0, 0, center_start:center_end, 0] == 1)


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
