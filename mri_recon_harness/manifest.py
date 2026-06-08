from __future__ import annotations

import csv
from pathlib import Path

import h5py

from mri_recon_harness.config import ProgramConfig


def write_manifests(config: ProgramConfig, manifest_dir: str | Path = "manifests") -> tuple[Path, Path]:
    out_dir = Path(manifest_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_path = out_dir / "train_manifest.tsv"
    val_path = out_dir / "val_manifest.tsv"
    _write_split_manifest(config.fastmri_root / "multicoil_train", train_path, config.train_files)
    _write_split_manifest(config.fastmri_root / "multicoil_val", val_path, config.val_files)
    return train_path, val_path


def _write_split_manifest(split_dir: Path, output_path: Path, max_files: int) -> None:
    files = sorted(split_dir.glob("*.h5"))
    if not files:
        raise FileNotFoundError(f"No .h5 files found in {split_dir}")
    if len(files) < max_files:
        raise ValueError(f"Requested {max_files} files from {split_dir}, found {len(files)}")

    rows: list[tuple[str, int]] = []
    for file_path in files[:max_files]:
        with h5py.File(file_path, "r") as hf:
            if "kspace" not in hf or "reconstruction_rss" not in hf:
                raise ValueError(f"Missing kspace or reconstruction_rss in {file_path}")
            num_slices = int(hf["kspace"].shape[0])
        for slice_idx in range(num_slices):
            rows.append((str(file_path), slice_idx))

    if not rows:
        raise ValueError(f"No slices found in selected files from {split_dir}")

    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["file_path", "slice_num"])
        writer.writerows(rows)


def read_manifest(path: str | Path) -> list[tuple[Path, int]]:
    rows: list[tuple[Path, int]] = []
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            rows.append((Path(row["file_path"]), int(row["slice_num"])))
    if not rows:
        raise ValueError(f"Manifest is empty: {path}")
    return rows
