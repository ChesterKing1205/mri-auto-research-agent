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

## Setup

Before starting a new Auto Research run:

1. Check that the repository is clean:

```bash
git status --short
```

Only ignored files such as `.venv/`, `results.tsv`, `run.log`, `manifests/`, and `outputs/` may be present.

2. Create or switch to the dedicated local branch:

```bash
git checkout -b autoresearch/mri-recon-psnr
```

If the branch already exists, switch to it only after confirming the working tree is clean:

```bash
git switch autoresearch/mri-recon-psnr
```

3. Read the in-scope project files before proposing trial ideas:

```text
README.md
program.md
mri_recon_project/
```

4. Verify the data root exists. If deterministic manifests are missing, run:

```bash
uv run prepare.py
```

If `prepare.py` fails because data or environment setup is missing, stop and report the setup issue. Do not modify code to bypass missing data.

5. If `results.tsv` does not exist, initialize it with the header row shown in the Logging section. Keep it untracked.

## Git and Remote Repository Policy

You may use local git commands to inspect status, create a local Auto Research branch, create commits, amend the current trial commit when fixing a simple crash, and reset failed local trials as described in this program.

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

Never commit generated or local-only files:

```text
results.tsv
run.log
manifests/
outputs/
.venv/
```

Only run destructive reset commands on a branch whose name starts with `autoresearch/`. Never run `git reset --hard` on `main`.

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
epochs_per_round: 30
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

## Evaluation Standard

The goal is to maximize PSNR under the fixed file split and training budget. Since each round uses the same manifests, epochs, acceleration, center fraction, and metric code, compare trials directly.

Use this decision order:

1. A higher PSNR is the main reason to keep a trial.
2. If PSNR is unchanged within noise, keep only if the code is clearly simpler or more robust.
3. If PSNR improves by a tiny amount but adds fragile, hard-to-maintain complexity, prefer discarding it.
4. SSIM, NMSE, and validation loss are supporting evidence only. Do not keep a trial with worse PSNR merely because an auxiliary metric improved.

## Commands

Prepare deterministic manifests:

```bash
uv run prepare.py
```

Run one fixed train+validation experiment:

```bash
timeout 30m uv run train.py > run.log 2>&1
run_status=$?
```

Read metrics:

```bash
grep -E "^(primary_metric|psnr|ssim|nmse|val_loss):" run.log
```

## Logging

When an experiment finishes, append one tab-separated row to `results.tsv`. Use tabs, not commas.

`results.tsv` must stay untracked. Do not commit it.

Columns:

```text
timestamp	start_commit	trial_commit	attempt	hypothesis	change_summary	psnr	ssim	nmse	val_loss	effective	decision
```

Column meanings:

- `timestamp`: local timestamp when the row is written.
- `start_commit`: commit where the trial started; this is the previous best state.
- `trial_commit`: commit that contains the trial code; for baseline, use `start_commit`.
- `attempt`: one short sentence describing what this trial changes.
- `hypothesis`: why this change might improve PSNR.
- `change_summary`: concrete summary of files/functions changed.
- `psnr`, `ssim`, `nmse`, `val_loss`: parsed values from `run.log`.
- `effective`: `yes` if the attempt improves PSNR over the previous best, otherwise `no`.
- `decision`: `baseline`, `keep`, `discard`, `crash`, or `timeout`.

For crashes and timeouts, use:

```text
psnr=0.000000
ssim=0.000000
nmse=inf
val_loss=inf
effective=no
```

## The Experiment Loop

This loop follows the `karpathy/autoresearch` git pattern: the branch tip always represents the current best known state. A failed trial is committed for inspection, recorded in `results.tsv`, and then removed by resetting back to the commit where the trial started.

LOOP FOREVER until the human interrupts:

0. Ensure you are on the dedicated local branch and the tracked working tree is clean:

```bash
git branch --show-current
git status --short
```

Only ignored files such as `results.tsv`, `run.log`, `manifests/`, and `outputs/` may be present.

If the current branch does not start with `autoresearch/`, stop. Do not run the loop or reset history on `main`.

1. If the baseline has not been recorded, run the baseline without editing code:

```bash
start_commit=$(git rev-parse HEAD)
uv run prepare.py
timeout 30m uv run train.py > run.log 2>&1
grep -E "^(primary_metric|psnr|ssim|nmse|val_loss):" run.log
```

Record the baseline in `results.tsv` with `decision=baseline`, `effective=yes`, and `trial_commit=start_commit`.

If the baseline crashes or metrics cannot be parsed, this is a setup failure. Inspect `tail -n 80 run.log`, report the problem, and stop. Do not edit research code to hide a broken baseline.

2. At the start of each new trial, save the current branch tip:

```bash
start_commit=$(git rev-parse HEAD)
```

This commit is the current best state before the trial. Compare every new result against the best PSNR in `results.tsv`, not merely the previous attempted row.

3. Make one small research change under `mri_recon_project/` only.

4. Commit the trial before running it:

```bash
git add mri_recon_project/
git commit -m "Trial: <short attempt description>"
trial_commit=$(git rev-parse HEAD)
```

Do not include `results.tsv`, `run.log`, `manifests/`, or `outputs/` in the commit.

5. Run the experiment with redirected output:

```bash
timeout 30m uv run train.py > run.log 2>&1
run_status=$?
```

The training budget is 15 minutes. The outer `timeout 30m` is a guard for hangs, slow data stalls, or deadlocks. If `run_status` is `124`, mark the trial as `timeout`, append a row to `results.tsv`, and reset to `start_commit`.

6. Read PSNR and auxiliary metrics:

```bash
grep -E "^(primary_metric|psnr|ssim|nmse|val_loss):" run.log
```

If the grep output is empty or the primary metric cannot be parsed, treat the run as a crash. Inspect the error:

```bash
tail -n 80 run.log
```

Crash handling:

- If the crash is a simple implementation error caused by the current trial, fix it under `mri_recon_project/`, amend the trial commit with `git commit --amend --no-edit`, update `trial_commit`, and rerun.
- Make at most three crash-fix attempts for one trial.
- If the idea is fundamentally broken, records invalid metrics, needs new dependencies, changes frozen files, or still crashes after the fix attempts, append a `decision=crash` row and reset to `start_commit`.

7. Append the trial to `results.tsv` before any reset.

8. If PSNR improves over the best PSNR so far, keep the trial commit. Record `decision=keep` and `effective=yes`. The branch tip is now the new best state.

9. If PSNR is equal or worse, record `decision=discard` and `effective=no`, then discard the trial code and return to the previous best state:

```bash
git reset --hard "$start_commit"
```

Do not reset `results.tsv`; it is ignored by git and should keep the full experiment history.

Before running `git reset --hard`, verify that:

- `git branch --show-current` starts with `autoresearch/`.
- `start_commit` was captured at the beginning of the current trial.
- Any uncommitted tracked changes belong only to the current failed trial.

If unexpected tracked changes appear outside `mri_recon_project/`, stop and report them instead of resetting.

## Research Discipline

- Change one idea per round.
- Prefer small edits with clear hypotheses.
- Do not change the data split, metric computation, training entrypoint, or harness.
- Do not optimize for validation loss if PSNR gets worse.
- Treat failures as useful evidence and record them.
- Keep going without asking whether to continue once the loop has started.
- Keep the `ResearchModule` public methods compatible with the harness: `train_batch`, `validate_batch`, and `configure_optimizers`.
