from __future__ import annotations

from pathlib import Path


EDITABLE_PREFIX = "mri_recon_project/"


def find_frozen_changes(changed_paths: list[str]) -> list[str]:
    frozen: list[str] = []
    for raw_path in changed_paths:
        path = raw_path.replace("\\", "/")
        if path == "results.tsv":
            continue
        if path.startswith(EDITABLE_PREFIX):
            continue
        frozen.append(path)
    return frozen


def assert_only_project_changed(changed_paths: list[str]) -> None:
    frozen = find_frozen_changes(changed_paths)
    if frozen:
        joined = "\n".join(f"- {path}" for path in frozen)
        raise RuntimeError(f"Frozen files were modified:\n{joined}")


def read_changed_paths_file(path: str | Path) -> list[str]:
    return [line.strip() for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]

