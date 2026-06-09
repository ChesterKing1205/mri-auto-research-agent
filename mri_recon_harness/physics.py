from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from fastmri import complex_abs as fastmri_complex_abs
from fastmri import complex_conj as fastmri_complex_conj
from fastmri import complex_mul as fastmri_complex_mul
from fastmri import fft2c as fastmri_fft2c
from fastmri import ifft2c as fastmri_ifft2c
from fastmri import rss_complex as fastmri_rss_complex


def to_tensor_complex_last(array) -> torch.Tensor:
    tensor = torch.from_numpy(array)
    if torch.is_complex(tensor):
        tensor = torch.view_as_real(tensor)
    return tensor.float()


def complex_last_to_complex(x: torch.Tensor) -> torch.Tensor:
    return torch.view_as_complex(x.contiguous())


def complex_to_complex_last(x: torch.Tensor) -> torch.Tensor:
    return torch.view_as_real(x).float()


def fft2c(image: torch.Tensor) -> torch.Tensor:
    return fastmri_fft2c(image, norm="ortho")


def ifft2c(kspace: torch.Tensor) -> torch.Tensor:
    return fastmri_ifft2c(kspace, norm="ortho")


def complex_abs(x: torch.Tensor) -> torch.Tensor:
    return fastmri_complex_abs(x)


def rss(image: torch.Tensor, dim: int = 1) -> torch.Tensor:
    return fastmri_rss_complex(image, dim=dim)


def center_crop_image(image: torch.Tensor, shape: tuple[int, int]) -> torch.Tensor:
    h, w = image.shape[-3], image.shape[-2]
    target_h, target_w = shape
    if h < target_h or w < target_w:
        pad_h = max(0, target_h - h)
        pad_w = max(0, target_w - w)
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        image_ch = image.permute(*range(image.ndim - 3), -1, -3, -2)
        image_ch = F.pad(image_ch, (pad_left, pad_right, pad_top, pad_bottom), mode="reflect")
        image = image_ch.permute(*range(image_ch.ndim - 3), -2, -1, -3).contiguous()
        h, w = image.shape[-3], image.shape[-2]
    top = (h - target_h) // 2
    left = (w - target_w) // 2
    return image[..., top : top + target_h, left : left + target_w, :]


def standardize_kspace(kspace: torch.Tensor, shape: tuple[int, int] = (320, 320)) -> torch.Tensor:
    image = ifft2c(kspace)
    cropped = center_crop_image(image, shape)
    return fft2c(cropped)


def complex_last_to_channels(x: torch.Tensor) -> torch.Tensor:
    if x.ndim != 5 or x.shape[-1] != 2:
        raise ValueError("Expected complex-last tensor with shape (B,C,H,W,2)")
    b, c, h, w, _ = x.shape
    return x.permute(0, 4, 1, 2, 3).reshape(b, 2 * c, h, w)


def channels_to_complex_last(x: torch.Tensor, coils: int = 1) -> torch.Tensor:
    if x.ndim != 4 or x.shape[1] != coils * 2:
        raise ValueError(f"Expected channel-first tensor with {coils * 2} channels")
    b, _, h, w = x.shape
    return x.view(b, 2, coils, h, w).permute(0, 2, 3, 4, 1).contiguous()


def center_crop_real(image: torch.Tensor, shape: tuple[int, int]) -> torch.Tensor:
    h, w = image.shape[-2], image.shape[-1]
    target_h, target_w = shape
    if h < target_h or w < target_w:
        pad_h = max(0, target_h - h)
        pad_w = max(0, target_w - w)
        image = F.pad(image, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2))
        h, w = image.shape[-2], image.shape[-1]
    top = (h - target_h) // 2
    left = (w - target_w) // 2
    return image[..., top : top + target_h, left : left + target_w]


def make_equispaced_mask(width: int, acceleration: int, center_fraction: float) -> torch.Tensor:
    num_low_freqs = max(1, round(width * center_fraction))
    target_samples = max(num_low_freqs, round(width / acceleration))
    mask = torch.zeros(width, dtype=torch.float32)
    pad = (width - num_low_freqs + 1) // 2
    mask[pad : pad + num_low_freqs] = 1.0
    remaining = target_samples - num_low_freqs
    if remaining <= 0:
        return mask

    candidates = torch.cat([torch.arange(0, pad), torch.arange(pad + num_low_freqs, width)])
    positions = torch.linspace(0, len(candidates) - 1, remaining).round().long()
    mask[candidates[positions]] = 1.0
    return mask


def make_gaussian_mask(width: int, total_samples: int, acs: int, seed: int | None = None) -> torch.Tensor:
    """Return a variable-density phase-encoding mask shaped for fastMRI broadcasting."""
    if acs > total_samples:
        raise ValueError("acs must be <= total_samples")
    if total_samples > width:
        raise ValueError("total_samples must be <= width")

    rng = np.random.default_rng(seed)
    mask = np.zeros(width, dtype=np.float32)
    start = width // 2 - acs // 2
    end = start + acs
    mask[start:end] = 1.0

    extra_samples = total_samples - int(mask.sum())
    if extra_samples > 0:
        x = np.arange(width)
        pdf = np.exp(-((x - width // 2) ** 2) / (2 * (width / 10.0) ** 2)).astype(np.float64)
        pdf += 0.02
        pdf[mask == 1] = 0
        pdf /= pdf.sum()
        mask[rng.choice(width, extra_samples, replace=False, p=pdf)] = 1.0

    return torch.from_numpy(mask.reshape(1, 1, width, 1))


def mask_center(x: torch.Tensor, mask_from: int, mask_to: int) -> torch.Tensor:
    mask = torch.zeros_like(x)
    mask[:, :, :, mask_from:mask_to] = x[:, :, :, mask_from:mask_to]
    return mask


def batched_mask_center(x: torch.Tensor, mask_from: torch.Tensor, mask_to: torch.Tensor) -> torch.Tensor:
    if mask_from.shape != mask_to.shape:
        raise ValueError("mask_from and mask_to must match shapes")
    if mask_from.ndim != 1:
        raise ValueError("mask_from and mask_to must be 1D")
    if mask_from.shape[0] == 1:
        return mask_center(x, int(mask_from.item()), int(mask_to.item()))
    if x.shape[0] != mask_from.shape[0]:
        raise ValueError("mask_from and mask_to must have batch_size length")

    mask = torch.zeros_like(x)
    for i, (start, end) in enumerate(zip(mask_from, mask_to)):
        mask[i, :, :, int(start.item()) : int(end.item())] = x[
            i, :, :, int(start.item()) : int(end.item())
        ]
    return mask


def get_pad_and_num_low_freqs(
    mask: torch.Tensor, num_low_frequencies: Optional[int] = None
) -> tuple[torch.Tensor, torch.Tensor]:
    if num_low_frequencies is None or num_low_frequencies == 0:
        squeezed_mask = mask[:, 0, 0, :, 0].to(torch.int8)
        center = squeezed_mask.shape[1] // 2
        left = torch.argmin(squeezed_mask[:, :center].flip(1), dim=1)
        right = torch.argmin(squeezed_mask[:, center:], dim=1)
        num_low_frequencies_tensor = torch.max(2 * torch.min(left, right), torch.ones_like(left))
    else:
        num_low_frequencies_tensor = num_low_frequencies * torch.ones(
            mask.shape[0], dtype=mask.dtype, device=mask.device
        )
    pad = (mask.shape[-2] - num_low_frequencies_tensor + 1) // 2
    return pad, num_low_frequencies_tensor


def estimate_sens_maps(masked_kspace: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Estimate coil sensitivity maps from the sampled ACS region."""
    pad, num_low_freqs = get_pad_and_num_low_freqs(mask)
    center_kspace = batched_mask_center(masked_kspace, pad, pad + num_low_freqs)
    coil_images = ifft2c(center_kspace)
    rss_image = rss(coil_images, dim=1).unsqueeze(1).unsqueeze(-1)
    return coil_images / rss_image.clamp_min(1e-12)


def sens_reduce(kspace: torch.Tensor, sens_maps: torch.Tensor) -> torch.Tensor:
    """Convert multicoil k-space to a single-coil complex image."""
    b, c, h, w, _ = kspace.shape
    image = ifft2c(kspace)
    return fastmri_complex_mul(image, fastmri_complex_conj(sens_maps)).view(b, 1, c, h, w, 2).sum(dim=2)


def sens_expand(image: torch.Tensor, sens_maps: torch.Tensor) -> torch.Tensor:
    """Convert a single-coil complex image to multicoil k-space."""
    _, coils, _, _, _ = sens_maps.shape
    return fft2c(fastmri_complex_mul(image.repeat_interleave(coils, dim=1), sens_maps))


def hard_data_consistency(
    pred_kspace: torch.Tensor,
    masked_kspace: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    return torch.where(mask.to(dtype=torch.bool, device=pred_kspace.device), masked_kspace, pred_kspace)
