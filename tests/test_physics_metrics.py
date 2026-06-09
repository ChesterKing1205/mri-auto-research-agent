from __future__ import annotations

import torch
from fastmri import complex_abs as fastmri_complex_abs
from fastmri import rss_complex as fastmri_rss_complex

from mri_recon_harness.metrics import compute_metrics
from mri_recon_harness.physics import (
    channels_to_complex_last,
    complex_abs,
    complex_last_to_channels,
    estimate_sens_maps,
    fft2c,
    hard_data_consistency,
    ifft2c,
    make_equispaced_mask,
    make_gaussian_mask,
    rss,
    sens_expand,
    sens_reduce,
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
    x = torch.randn(2, 3, 16, 16, 2)
    channels = complex_last_to_channels(x)
    restored = channels_to_complex_last(channels, coils=3)
    assert restored.shape == x.shape
    assert torch.allclose(x, restored)


def test_complex_abs_and_rss_match_fastmri_helpers():
    x = torch.randn(2, 4, 16, 16, 2)
    assert torch.allclose(complex_abs(x), fastmri_complex_abs(x))
    assert torch.allclose(rss(x, dim=1), fastmri_rss_complex(x, dim=1))


def test_equispaced_mask_matches_requested_acceleration():
    mask = make_equispaced_mask(width=320, acceleration=4, center_fraction=0.08)
    sampled = int(mask.sum().item())
    assert sampled == 80
    assert 320 / sampled == 4.0
    center = round(320 * 0.08)
    pad = (320 - center + 1) // 2
    assert torch.all(mask[pad : pad + center] == 1)


def test_gaussian_mask_matches_reference_shape_and_acs():
    mask = make_gaussian_mask(width=320, total_samples=80, acs=24, seed=123)
    assert mask.shape == (1, 1, 320, 1)
    assert int(mask.sum().item()) == 80
    start = 320 // 2 - 24 // 2
    assert torch.all(mask[0, 0, start : start + 24, 0] == 1)


def test_sensitivity_reduce_expand_and_data_consistency_shapes():
    kspace = torch.randn(2, 4, 32, 32, 2)
    mask = make_gaussian_mask(width=32, total_samples=16, acs=8, seed=123).to(kspace).unsqueeze(0)
    mask = mask.repeat(kspace.shape[0], 1, 1, 1, 1)
    masked_kspace = kspace * mask

    sens_maps = estimate_sens_maps(masked_kspace, mask)
    reduced = sens_reduce(masked_kspace, sens_maps)
    expanded = sens_expand(reduced, sens_maps)
    dc = hard_data_consistency(expanded, masked_kspace, mask)

    assert sens_maps.shape == kspace.shape
    assert reduced.shape == (2, 1, 32, 32, 2)
    assert expanded.shape == kspace.shape
    assert torch.allclose(dc * mask, masked_kspace * mask)


def test_metrics_are_finite_for_equal_images():
    target = torch.rand(2, 1, 32, 32)
    metrics = compute_metrics(target, target)
    assert torch.isfinite(metrics["psnr"])
    assert torch.isfinite(metrics["ssim"])
    assert torch.isfinite(metrics["nmse"])
    assert metrics["nmse"].item() == 0.0
