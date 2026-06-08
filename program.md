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

## Git and Remote Repository Policy

You may use local git commands to inspect status, create a new local Auto Research branch, create commits, and reset failed local trials as described in this program.

Do not run remote repository operations unless the user explicitly asks in the current Codex session. This includes:

```text
git remote add
git remote set-url
git push
git pull
git fetch
git force-push
gh repo create
gh repo delete
```

Do not change the remote URL, delete branches, rewrite published history, or force push. Auto Research should manage local experiment commits only.

Before starting research, create a local branch from the current clean baseline:

```bash
git checkout -b autoresearch/mri-recon-psnr
```

If that branch already exists, switch to it only after checking that the working tree is clean.

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
train_files: 10
val_files: 3
epochs_per_round: 10
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

This loop follows the `karpathy/autoresearch` git pattern: the branch tip always represents the current best known state. A failed trial is committed for inspection, recorded in `results.tsv`, and then removed by resetting back to the commit where the trial started.

0. Ensure the working tree is clean before every trial:

```bash
git status --short
```

Only ignored files such as `results.tsv`, `run.log`, `manifests/`, and `outputs/` may be present.

1. If `results.tsv` does not exist, run the baseline without editing code:

```bash
uv run prepare.py
uv run train.py > run.log 2>&1
```

Record the baseline in `results.tsv` with `decision=baseline`, `effective=yes`, and `start_commit` equal to the current commit.

2. At the start of each new trial, save the current branch tip:

```bash
start_commit=$(git rev-parse HEAD)
```

This commit is the current best state before the trial.

3. Make one small research change under `mri_recon_project/` only.

4. Run:

```bash
uv run train.py > run.log 2>&1
```

5. Read PSNR and auxiliary metrics from `run.log`.

6. Commit the trial:

```bash
git add mri_recon_project/
git commit -m "Trial: <short attempt description>"
trial_commit=$(git rev-parse HEAD)
```

7. Append the trial to `results.tsv` before any reset.

8. If PSNR improves over the best PSNR so far, keep the trial commit. The branch tip is now the new best state.

9. If PSNR does not improve, discard the trial code and return to the previous best state:

```bash
git reset --hard "$start_commit"
```

Do not reset `results.tsv`; it is ignored by git and should keep the full experiment history.

Suggested `results.tsv` columns:

```text
timestamp	start_commit	trial_commit	attempt	hypothesis	change_summary	psnr	ssim	nmse	val_loss	effective	decision
```

Column meanings:

- `attempt`: one short sentence describing what this trial changes.
- `effective`: `yes` if the attempt improves PSNR over the previous best, otherwise `no`.
- `decision`: `baseline`, `keep`, or `discard`.

## Research Discipline

- Change one idea per round.
- Prefer small edits with clear hypotheses.
- Do not change the data split, metric computation, training entrypoint, or harness.
- Do not optimize for validation loss if PSNR gets worse.
- Treat failures as useful evidence and record them.
- Keep the `ResearchModule` public methods compatible with the harness: `train_batch`, `validate_batch`, and `configure_optimizers`.
