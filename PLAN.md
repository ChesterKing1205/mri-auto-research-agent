# MRI Auto Research Agent Plan

## Summary

Build a fresh Auto Research environment for fastMRI multicoil MRI reconstruction. The project follows the core control flow of `karpathy/autoresearch`: Codex is the external research agent, reads `program.md`, edits only the research project, runs fixed experiment commands, reads a single primary metric, and uses git keep/discard to continue from the best result.

This project is not Deep Research and not a full AI Scientist. Deep Research focuses on search and synthesis reports. AI Scientist targets end-to-end paper generation and review. This project focuses on controlled code experiments for MRI reconstruction.

## Structure

```text
mri-auto-research-agent/
  README.md
  PLAN.md
  program.md
  prepare.py
  train.py
  pyproject.toml

  mri_recon_harness/      # fixed experiment judge
  mri_recon_project/      # Codex-editable research code
  tests/
```

## Fixed Rules

- Codex may edit only `mri_recon_project/`.
- Codex must not edit `program.md`, `prepare.py`, `train.py`, or `mri_recon_harness/`.
- `prepare.py` checks the environment and creates deterministic train/val file manifests expanded to all slices in selected files.
- `train.py` runs one fixed train+validation experiment and prints stable `key: value` metrics.
- The primary metric is validation PSNR, maximized.
- SSIM, NMSE, and validation loss are auxiliary metrics.
- Image data is not cached. Each sample is read from the original fastMRI HDF5 file.
- Dependencies are managed by `uv`; Codex must not add Python dependencies during research loops.

## Harness Responsibilities

- Own fastMRI multicoil loading, deterministic subset manifests, masks, target images, Lightning Trainer, metric computation, TensorBoard logging, and checkpoints.
- Keep k-space and image physics tensors in complex-last format outside the neural network boundary.
- Use `reconstruction_rss` from fastMRI HDF5 as the target image.
- Compute PSNR, SSIM, and NMSE in the harness, not in the editable project.
- Call `mri_recon_project.build_research_module(config)` and delegate model/loss/optimizer details to the editable project.

## Project Responsibilities

- Provide `build_research_module(config)`.
- Return a `ResearchModule` object with `train_batch(batch)`, `validate_batch(batch)`, and `configure_optimizers()`.
- `train_batch` and `validate_batch` return a dict with `loss`, `pred_image`, and `logs`.
- `pred_image` must be a restored-scale magnitude image aligned to `target_image`.
- The project may change model architecture, loss, training logic, validation logic, optimizer, scheduler, and research hyperparameters.

## Auto Research Loop

1. Human starts Codex in this repository.
2. Codex reads `program.md`.
3. Codex runs `uv run prepare.py`.
4. Codex runs the baseline with `uv run train.py`.
5. Codex records baseline metrics in local `results.tsv`, including `attempt` and `effective` columns.
6. Codex records the current branch tip as `start_commit`.
7. Codex makes one small change under `mri_recon_project/`.
8. Codex runs `uv run train.py`.
9. Codex commits every trial.
10. If PSNR improves, Codex keeps the commit as the new branch tip and best state.
11. If PSNR does not improve, Codex records the trial and resets back to `start_commit`.

## Verification

- Unit tests cover physics helpers, metrics, manifest generation, dataset fields, project API, and stdout metric formatting.
- A smoke run should complete with a very small subset before any long experiment.
