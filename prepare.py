from __future__ import annotations

from mri_recon_harness.config import load_program_config
from mri_recon_harness.manifest import write_manifests


def main() -> None:
    config = load_program_config("program.md")
    train_path, val_path = write_manifests(config)
    print(f"train_manifest: {train_path}")
    print(f"val_manifest: {val_path}")


if __name__ == "__main__":
    main()
