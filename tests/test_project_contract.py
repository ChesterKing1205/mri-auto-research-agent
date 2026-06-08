from __future__ import annotations

import torch

from mri_recon_project import build_research_module


def test_research_module_contract_outputs_loss_pred_logs():
    module = build_research_module({"base_channels": 4})
    batch = {
        "zero_filled_image": torch.rand(1, 1, 32, 32),
        "target_image": torch.rand(1, 1, 32, 32),
    }
    output = module.train_batch(batch)
    assert set(output) == {"loss", "pred_image", "logs"}
    assert output["loss"].ndim == 0
    assert output["pred_image"].shape == batch["target_image"].shape
    assert isinstance(output["logs"], dict)
    assert module.configure_optimizers() is not None
