# MRI Auto Research Agent

This is a fresh Auto Research environment for fastMRI multicoil MRI reconstruction.

The project follows the control flow of `karpathy/autoresearch`: Codex is the external research agent. Codex reads `program.md`, edits only `mri_recon_project/`, runs fixed experiments through `prepare.py` and `train.py`, reads validation PSNR, then keeps or discards changes with git.

## Auto Research vs Deep Research vs AI Scientist

- Auto Research: controlled code experiments around a fixed metric and fixed judge.
- Deep Research: search, source synthesis, and report writing.
- AI Scientist: broad automation from idea generation to experiments, paper writing, and review.

This project implements Auto Research for MRI reconstruction, not web research and not full paper generation.

## Layout

```text
program.md              # rules for Codex
prepare.py              # fixed manifest/environment preparation
train.py                # fixed train+validation entrypoint
mri_recon_harness/      # fixed data, Lightning, metrics, checkpoints
mri_recon_project/      # Codex-editable model/loss/optimizer code
```

Codex may edit only `mri_recon_project/`.

The editable project must keep the harness-facing API stable:

```text
build_research_module(config)
ResearchModule.train_batch(batch)
ResearchModule.validate_batch(batch)
ResearchModule.configure_optimizers()
```

## Install

```bash
cd ~/projects/mri-auto-research-agent
uv sync
```

If you already use the `kspace` conda environment:

```bash
/home/chesterking/miniconda3/envs/kspace/bin/python -m pytest
```

## Prepare

Edit `program.md` if you need to change the data root or small-run budget.

```bash
uv run prepare.py
```

The prepare step writes deterministic manifests under `manifests/` by selecting a fixed number of train/val HDF5 files and expanding all slices in those files. It does not cache image data.

## Run One Experiment

```bash
uv run train.py > run.log 2>&1
grep "^primary_metric:" run.log
grep "^psnr:" run.log
grep "^ssim:" run.log
grep "^nmse:" run.log
grep "^val_loss:" run.log
```

Training artifacts are written under `outputs/version_N` by Lightning's TensorBoard logger.

## Start Auto Research with Codex

```bash
cd ~/projects/mri-auto-research-agent
codex
```

Then ask Codex:

```text
Read program.md and start the Auto Research loop. Run the baseline first, then improve only mri_recon_project/.
```

Codex should maintain local `results.tsv`, commit every trial, keep commits that improve PSNR, and reset to the best commit when PSNR does not improve.

## WSL GitHub Setup

Create a local repository:

```bash
git init
git add .
git commit -m "Initial MRI Auto Research environment"
```

Create an empty GitHub repository in the browser, then connect it:

```bash
git branch -M main
git remote add origin git@github.com:<your-user>/mri-auto-research-agent.git
git push -u origin main
```

If SSH is not configured, use the HTTPS remote shown by GitHub instead.

## Tests

```bash
uv run pytest
```

The tests cover manifest creation, physics helpers, metric calculation, the project API, and stable metric output.
