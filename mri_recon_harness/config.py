from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProgramConfig:
    fastmri_root: Path
    train_files: int
    val_files: int
    epochs_per_round: int
    batch_size: int
    max_minutes_per_round: int
    acceleration: int
    center_fraction: float
    acs: int
    num_workers: int = 0


DEFAULTS = {
    "fastMRI_root": "/mnt/d/fastmri/brain/T1",
    "train_files": "1",
    "val_files": "1",
    "epochs_per_round": "1",
    "batch_size": "1",
    "max_minutes_per_round": "15",
    "acceleration": "4",
    "center_fraction": "0.08",
    "acs": "24",
}


def _extract_value(text: str, key: str) -> str:
    pattern = re.compile(rf"^\s*{re.escape(key)}\s*:\s*(.+?)\s*$", re.MULTILINE)
    match = pattern.search(text)
    if match:
        return match.group(1).strip()
    return DEFAULTS[key]


def load_program_config(path: str | Path) -> ProgramConfig:
    text = Path(path).read_text(encoding="utf-8")
    root = Path(_extract_value(text, "fastMRI_root")).expanduser()
    config = ProgramConfig(
        fastmri_root=root,
        train_files=int(_extract_value(text, "train_files")),
        val_files=int(_extract_value(text, "val_files")),
        epochs_per_round=int(_extract_value(text, "epochs_per_round")),
        batch_size=int(_extract_value(text, "batch_size")),
        max_minutes_per_round=int(_extract_value(text, "max_minutes_per_round")),
        acceleration=int(_extract_value(text, "acceleration")),
        center_fraction=float(_extract_value(text, "center_fraction")),
        acs=int(_extract_value(text, "acs")),
    )
    _validate_config(config)
    return config


def _validate_config(config: ProgramConfig) -> None:
    if not config.fastmri_root.exists():
        raise FileNotFoundError(f"fastMRI_root does not exist: {config.fastmri_root}")
    for split in ("multicoil_train", "multicoil_val"):
        split_dir = config.fastmri_root / split
        if not split_dir.exists():
            raise FileNotFoundError(f"Missing fastMRI split directory: {split_dir}")
    if config.train_files < 1 or config.val_files < 1:
        raise ValueError("train_files and val_files must be positive")
    if config.batch_size < 1:
        raise ValueError("batch_size must be positive")
    if config.epochs_per_round < 1:
        raise ValueError("epochs_per_round must be positive")
    if config.acceleration < 2:
        raise ValueError("acceleration must be >= 2")
    if not 0 < config.center_fraction < 1:
        raise ValueError("center_fraction must be in (0, 1)")
    width = 320
    total_samples = round(width / config.acceleration)
    if config.acs < 1:
        raise ValueError("acs must be positive")
    if config.acs > total_samples:
        raise ValueError("acs must be <= width / acceleration")
