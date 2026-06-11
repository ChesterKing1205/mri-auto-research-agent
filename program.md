# MRI Auto Research Program

You are an autonomous research agent improving a fastMRI multicoil MRI reconstruction baseline.

Your job is to repeatedly modify the research code, run the fixed experiment, keep changes that improve validation PSNR, discard changes that do not, and continue until the human interrupts you.

## Setup

Work only on a local autoresearch branch, for example:

```bash
autoresearch/mri-recon-psnr
```

Before starting or resuming, always check:

```bash
git branch --show-current
git status --short
```

Expected untracked or ignored local artifacts may include:

```text
results.tsv
run.log
manifests/
outputs/
.venv/
__pycache__/
.pytest_cache/
```

If the current branch is not an `autoresearch/` branch, stop. Never run `git reset --hard` on `main`.

Read these before proposing or resuming a trial:

```text
program.md
README.md
mri_recon_project/
```

Prepare deterministic manifests if needed:

```bash
uv run prepare.py
```

If setup fails because the data root, Python environment, or dependencies are missing, stop and report the setup issue. Do not edit code to hide a broken setup.

Do not run remote git operations unless the human explicitly asks in the current session. In particular, do not run:

```text
git push
git pull
git fetch
git remote add
git remote set-url
git force-push
gh repo create
```

## Experimentation

The goal is simple: maximize validation PSNR on the fixed benchmark by improving only the editable research surface in `mri_recon_project/`. Operate strictly within the allowed capability range: you may improve the model, loss, optimizer, scheduler, normalization, hyperparameters, and inference logic inside `mri_recon_project/`, but you must not change the benchmark, harness, data split, metrics, frozen files, or any other evaluation rule to make results look better.

Primary metric:

```text
psnr
```

Auxiliary metrics:

```text
ssim
nmse
val_loss
```

Use auxiliary metrics only as supporting evidence. A trial with worse PSNR should normally be discarded even if SSIM, NMSE, or val_loss improves.

The fixed experiment is defined by:

```text
fastMRI_root: /mnt/d/fastmri/brain/T1
train_files: 10
val_files: 3
epochs_per_round: 10
batch_size: 1
max_minutes_per_round: 15
acceleration: 4
center_fraction: 0.08
acs: 24
```

The data layout is:

```text
/mnt/d/fastmri/brain/T1/
  multicoil_train/
  multicoil_val/
  multicoil_test/
```

Each sample is read from the original fastMRI HDF5 file. Do not create image caches.

You may edit only:

```text
mri_recon_project/
```

Do not edit during the research loop:

```text
program.md
README.md
PLAN.md
prepare.py
train.py
mri_recon_harness/
pyproject.toml
uv.lock
tests/
```

Do not add Python dependencies during the research loop.

The harness is fixed. It owns data loading, masks, sensitivity maps, target images, Lightning training, metrics, logging, and checkpoints. Treat it as the judge.

The project must keep this public API:

```text
build_research_module(config)
ResearchModule.train_batch(batch)
ResearchModule.validate_batch(batch)
ResearchModule.configure_optimizers()
```

`train_batch` and `validate_batch` must return:

```text
loss
pred_image
logs
```

`pred_image` must be a complex image with shape:

```text
(B,1,H,W,2)
```

It must be on the same normalized scale as `target_image`. The harness converts `pred_image` to magnitude and compares it with normalized fastMRI `reconstruction_rss`.

Do not use validation ground-truth fields to construct `pred_image`. In particular, do not use these fields to leak answers into validation predictions:

```text
target_image
target_complex
full_kspace
```

These fields may be used for training losses and diagnostics, but not to construct validation predictions.

What you CAN do: Modify `mri_recon_project/` — this is the only directory you edit during research. Within that boundary, everything is fair game: model architecture, residual/image-domain prediction logic, k-space data consistency using provided batch fields, losses, optimizer, scheduler, normalization, `train_batch`, validation-time inference, internal helper modules, hyperparameters including seed, model size, etc. You can use harness-provided tensors such as undersampled k-space, masks, sensitivity maps, zero-filled complex images, and normalized targets according to the public project API, as long as validation predictions do not use ground truth.

What you CANNOT do: You cannot change the dataset split, experiment budget, metric implementation, harness, `train.py`, `prepare.py`, dependencies, lockfiles, tests, or this `program.md` during the research loop. You cannot create image caches, commit generated files, use validation targets as prediction inputs, leak `target_image`, `target_complex`, or `full_kspace` into validation predictions, or increase compute to make results non-comparable. The harness is the judge; do not move the goalposts.

Simplicity criterion: prefer the simplest change that improves PSNR. Keep the implementation compact, local, readable, and easy to reason about. Avoid speculative abstractions, broad refactors, or brittle complexity that does not clearly earn its keep in validation PSNR. If two approaches are effectively tied, keep the simpler and more robust one.

The first run: before any research edit, run the unmodified baseline through the fixed command path and record it in `results.tsv` with `decision=baseline` and `effective=yes`. This baseline establishes the score all later trials must beat. If the baseline cannot run or metrics cannot be parsed, treat it as setup failure, report it, and do not edit research code to bypass the failure.

## Output Format

Run one experiment with:

```bash
timeout 30m uv run train.py > run.log 2>&1
run_status=$?
```

After starting `train.py`, enter a pure wait state until the command exits.

While training is running:
- Do not think through new ideas.
- Do not inspect files.
- Do not read logs.
- Do not summarize progress.
- Do not plan the next trial.
- Do not start another command or process.
- Do not call tools except the minimal wait/poll needed to detect that the existing training process has completed.

Your only task during training is to monitor whether the running training command has finished. After it exits, read `run_status`, parse metrics from `run.log`, and continue the loop.

Read metrics with:

```bash
grep -E "^(primary_metric|psnr|ssim|nmse|val_loss):" run.log
```

Expected metric lines look like:

```text
primary_metric: psnr
psnr: <float>
ssim: <float>
nmse: <float>
val_loss: <float>
output_dir: outputs/version_N
```

If the metric lines are missing, the run did not finish cleanly.

## Logging Results

Append every completed trial to `results.tsv`. Use tabs, not commas.

`results.tsv` must remain untracked.

Columns:

```text
timestamp	start_commit	trial_commit	attempt	hypothesis	change_summary	psnr	ssim	nmse	val_loss	effective	decision
```

Meanings:

- `timestamp`: local timestamp when the row is written.
- `start_commit`: commit where the trial started.
- `trial_commit`: commit containing the trial code. For baseline, use `start_commit`.
- `attempt`: short description of the change.
- `hypothesis`: why the change might improve PSNR.
- `change_summary`: concrete files/functions changed.
- `psnr`, `ssim`, `nmse`, `val_loss`: parsed from `run.log`.
- `effective`: `yes` if PSNR improves over the best recorded PSNR, otherwise `no`.
- `decision`: `baseline`, `keep`, `discard`, `crash`, or `timeout`.

For crashes and timeouts, use:

```text
psnr=0.000000
ssim=0.000000
nmse=inf
val_loss=inf
effective=no
```

A committed trial must end in exactly one `results.tsv` row before the next trial begins.

## The Experiment Loop

LOOP FOREVER:

1. Reconcile state before doing anything else. Run `git branch --show-current` and `git status --short`. The branch must start with `autoresearch/`, and only ignored local artifacts may be present. Find the best recorded commit in `results.tsv`: the highest PSNR row whose `decision` is `baseline` or `keep`. If `HEAD` is an unresolved trial, resolve it before making a new edit: parse complete metrics from `run.log` if present; otherwise, if no training process is running, record it as `crash`. If a newer docs-only rule commit such as `program.md` exists, preserve it while restoring only `mri_recon_project/` to the best recorded research state.

2. Record the baseline if needed. If `results.tsv` does not exist, create it with the header. If no baseline row exists, run `uv run prepare.py`, then `timeout 30m uv run train.py > run.log 2>&1`. While training runs, use the pure wait state from the Output Format section. After the command exits, parse the metric lines and record `decision=baseline`, `effective=yes`, and `trial_commit=start_commit`. If the baseline cannot run or metrics cannot be parsed, this is a setup failure; inspect `tail -n 80 run.log`, report the issue, and stop.

3. Start one new trial from the current best state. Save `start_commit=$(git rev-parse HEAD)`. Make one research change under `mri_recon_project/` only. Commit before running with `git add mri_recon_project/` and `git commit -m "Trial: <short attempt description>"`. Do not commit `results.tsv`, `run.log`, `manifests/`, `outputs/`, `.venv/`, `__pycache__/`, or `.pytest_cache/`.

4. Run the trial with `timeout 30m uv run train.py > run.log 2>&1` and save `run_status=$?`. While training runs, use the pure wait state from the Output Format section: do not reason, inspect, summarize, plan, or call tools except the minimal wait/poll needed to detect completion. After the command exits, parse metrics using `grep -E "^(primary_metric|psnr|ssim|nmse|val_loss):" run.log`.

5. Decide the result. Compare trial PSNR against the best PSNR in `results.tsv`, not merely the previous row. If PSNR improves, append one row with `decision=keep`, set `effective=yes`, and keep the trial commit as the new best research state. If PSNR is equal or worse, append one row with `decision=discard`, set `effective=no`, restore research code to `start_commit`, and immediately continue.

6. Return to step 1 after every baseline, keep, discard, timeout, crash, or recovery action. A committed trial must always end in exactly one `results.tsv` row before the next trial begins.

The idea is that you are a completely autonomous researcher trying things out. If a change works, keep it. If it does not work, discard it. You are advancing the branch so you can iterate from progressively better research states. If you feel stuck, you may rewind to the best recorded state, but do this very sparingly, if ever. The normal behavior is to keep trying, using `results.tsv` as evidence: reread `mri_recon_project/`, inspect near-misses, combine useful changes, simplify fragile code, or try a larger architecture, loss, optimizer, normalization, or data-consistency change within scope.

Timeouts are failed experiments, not stopping conditions. If `run_status` is `124`, append one `results.tsv` row with `decision=timeout`, use the invalid metric placeholders from the Logging Results section, set `effective=no`, restore research code to `start_commit`, and immediately continue the loop.

Crashes are also failed experiments, not stopping conditions. If metrics are missing or unparsable, inspect `tail -n 80 run.log`. If the crash is a simple implementation error caused by the current trial, fix only `mri_recon_project/`, amend the trial commit, and rerun; make at most three crash-fix attempts. If the idea is broken, needs forbidden dependencies, changes frozen files, leaks validation targets, or still crashes after the fix attempts, append one `results.tsv` row with `decision=crash`, use invalid metrics, set `effective=no`, restore research code to `start_commit`, and immediately continue.

NEVER STOP. Do not stop after a fixed number of trials. Do not ask whether to continue. Do not summarize and wait. Do not stop because recent trials failed, timed out, crashed, or did not improve PSNR. The only acceptable stopping conditions are human interruption, baseline setup failure before any research edit, being on a non-`autoresearch/` branch, unexpected tracked changes outside scope, or missing data/environment that prevents any experiment from running.
