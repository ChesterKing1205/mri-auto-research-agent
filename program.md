# MRI Auto Research Program

You are running an Auto Research loop for fastMRI multicoil MRI reconstruction.

## Goal

Maximize validation PSNR on the fixed validation subset.

## Editable Scope

You may edit only:

```text
mri_recon_project/
```

## Frozen Scope

Do not edit:

```text
program.md
prepare.py
train.py
mri_recon_harness/
pyproject.toml
tests/
```

Do not add Python dependencies during the research loop.

## Data

Default fastMRI root:

```text
/mnt/d/fastmri/brain/T1
```

Expected layout:

```text
multicoil_train/
multicoil_val/
multicoil_test/
```

Every sample is read from the original fastMRI HDF5 file. Do not create image caches.

## Experiment Budget

Fill or update these values before long research runs:

```text
fastMRI_root: /mnt/d/fastmri/brain/T1
train_files: 9
val_files: 3
epochs_per_round: 1
batch_size: 1
max_minutes_per_round: 15
acceleration: 4
center_fraction: 0.08
seed: 1337
```

## Primary Metric

Primary metric:

```text
psnr
```

Maximize this value.

Auxiliary metrics:

```text
ssim
nmse
val_loss
```

## Commands

Prepare deterministic manifests:

```bash
uv run prepare.py
```

Run one fixed train+validation experiment:

```bash
uv run train.py > run.log 2>&1
```

Read metrics:

```bash
grep "^primary_metric:" run.log
grep "^psnr:" run.log
grep "^ssim:" run.log
grep "^nmse:" run.log
grep "^val_loss:" run.log
```

## Research Loop

1. Run the baseline without editing any code.
2. Record baseline in local `results.tsv`.
3. Make one small research change under `mri_recon_project/`.
4. Run `uv run train.py > run.log 2>&1`.
5. Read PSNR and auxiliary metrics from `run.log`.
6. Commit the trial.
7. If PSNR improves over the best PSNR, keep the commit as the new best.
8. If PSNR does not improve, reset back to the best commit.
9. Append every attempt to `results.tsv`.

Suggested `results.tsv` columns:

```text
timestamp	commit	attempt	hypothesis	change_summary	psnr	ssim	nmse	val_loss	effective	decision
```

Column meanings:

- `attempt`: one short sentence describing what this trial changes.
- `effective`: `yes` if the attempt improves PSNR over the previous best, otherwise `no`.
- `decision`: `keep` or `discard`.

## Research Discipline

- Change one idea per round.
- Prefer small edits with clear hypotheses.
- Do not change the data split, metric computation, training entrypoint, or harness.
- Do not optimize for validation loss if PSNR gets worse.
- Treat failures as useful evidence and record them.
- Keep the `ResearchModule` public methods compatible with the harness: `train_batch`, `validate_batch`, and `configure_optimizers`.
