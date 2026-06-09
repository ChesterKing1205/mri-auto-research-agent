from __future__ import annotations

import torch

from mri_recon_harness.lightning_module import _prediction_magnitude


def test_prediction_magnitude_accepts_complex_single_coil_output():
    pred = torch.randn(2, 1, 16, 16, 2)
    magnitude = _prediction_magnitude(pred)
    assert magnitude.shape == (2, 1, 16, 16)
    assert torch.all(magnitude >= 0)


def test_prediction_magnitude_rejects_old_magnitude_output():
    pred = torch.randn(2, 1, 16, 16)
    try:
        _prediction_magnitude(pred)
    except ValueError as exc:
        assert "(B,1,H,W,2)" in str(exc)
    else:
        raise AssertionError("old magnitude pred_image output should be rejected")
