from __future__ import annotations

from mri_recon_harness.config import load_program_config
from mri_recon_harness.runner import run_experiment


def main() -> None:
    config = load_program_config("program.md")
    metrics = run_experiment(config)
    print("primary_metric: psnr")
    for key in ("psnr", "ssim", "nmse", "val_loss"):
        print(f"{key}: {metrics[key]:.6f}")
    print(f"output_dir: {metrics['output_dir']}")


if __name__ == "__main__":
    main()
