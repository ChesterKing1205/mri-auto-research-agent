from __future__ import annotations

from pathlib import Path

from mri_recon_harness.config import ProgramConfig
from mri_recon_harness import runner


def test_runner_seeds_lightning_from_program_config(monkeypatch, tmp_path):
    calls: list[tuple[int, bool]] = []

    def fake_seed_everything(seed: int, workers: bool = False):
        calls.append((seed, workers))

    class FakeLogger:
        log_dir = str(tmp_path / "outputs" / "version_0")

        def __init__(self, *args, **kwargs):
            pass

    class FakeCheckpoint:
        def __init__(self, *args, **kwargs):
            pass

    class FakeTrainer:
        callback_metrics = {
            "val_metrics/psnr": _Metric(1.0),
            "val_metrics/ssim": _Metric(1.0),
            "val_metrics/nmse": _Metric(0.0),
            "loss/val_loss": _Metric(0.0),
        }

        def __init__(self, *args, **kwargs):
            pass

        def fit(self, *args, **kwargs):
            pass

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(runner.pl, "seed_everything", fake_seed_everything)
    monkeypatch.setattr(runner, "write_manifests", lambda config: None)
    monkeypatch.setattr(runner, "FastMRIDataModule", lambda config: object())
    monkeypatch.setattr(runner, "HarnessLightningModule", lambda: object())
    monkeypatch.setattr(runner, "TensorBoardLogger", FakeLogger)
    monkeypatch.setattr(runner, "ModelCheckpoint", FakeCheckpoint)
    monkeypatch.setattr(runner.pl, "Trainer", FakeTrainer)

    config = ProgramConfig(
        fastmri_root=Path("/tmp/unused"),
        train_files=1,
        val_files=1,
        epochs_per_round=1,
        batch_size=1,
        max_minutes_per_round=15,
        acceleration=4,
        center_fraction=0.08,
        acs=24,
        seed=1234,
    )

    runner.run_experiment(config)

    assert calls == [(1234, True)]


class _Metric:
    def __init__(self, value: float) -> None:
        self.value = value

    def detach(self):
        return self

    def cpu(self):
        return self

    def __float__(self) -> float:
        return self.value
