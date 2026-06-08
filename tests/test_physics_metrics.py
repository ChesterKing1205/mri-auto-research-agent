from __future__ import annotations

import torch

from mri_recon_harness.metrics import compute_metrics
from mri_recon_harness.physics import (
    channels_to_complex_last,
    complex_last_to_channels,
    fft2c,
    ifft2c,
    standardize_kspace,
)


def test_fft_roundtrip_and_standardize_shape():
    kspace = torch.randn(1, 2, 640, 320, 2)
    standardized = standardize_kspace(kspace)
    assert standardized.shape == (1, 2, 320, 320, 2)
    image = ifft2c(standardized)
    recovered = fft2c(image)
    assert torch.allclose(standardized, recovered, atol=1e-5)


def test_complex_channel_roundtrip():
    x = torch.randn(2, 1, 16, 16, 2)
    channels = complex_last_to_channels(x)
    restored = channels_to_complex_last(channels, coils=1)
    assert restored.shape == x.shape
    assert torch.allclose(x, restored)


def test_metrics_are_finite_for_equal_images():
    target = torch.rand(2, 1, 32, 32)
    metrics = compute_metrics(target, target)
    assert torch.isfinite(metrics["psnr"])
    assert torch.isfinite(metrics["ssim"])
    assert torch.isfinite(metrics["nmse"])
    assert metrics["nmse"].item() == 0.0

