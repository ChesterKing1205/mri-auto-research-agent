from __future__ import annotations

import torch
import torch.nn.functional as F


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
    image_c = complex_last_to_complex(image)
    shifted = torch.fft.ifftshift(image_c, dim=(-2, -1))
    kspace = torch.fft.fft2(shifted, norm="ortho")
    kspace = torch.fft.fftshift(kspace, dim=(-2, -1))
    return complex_to_complex_last(kspace)


def ifft2c(kspace: torch.Tensor) -> torch.Tensor:
    kspace_c = complex_last_to_complex(kspace)
    shifted = torch.fft.ifftshift(kspace_c, dim=(-2, -1))
    image = torch.fft.ifft2(shifted, norm="ortho")
    image = torch.fft.fftshift(image, dim=(-2, -1))
    return complex_to_complex_last(image)


def complex_abs(x: torch.Tensor) -> torch.Tensor:
    return torch.sqrt((x**2).sum(dim=-1).clamp_min(1e-12))


def rss(image: torch.Tensor, dim: int = 1) -> torch.Tensor:
    return torch.sqrt((complex_abs(image) ** 2).sum(dim=dim).clamp_min(1e-12))


def center_crop_image(image: torch.Tensor, shape: tuple[int, int]) -> torch.Tensor:
    h, w = image.shape[-3], image.shape[-2]
    target_h, target_w = shape
    if h < target_h or w < target_w:
        raise ValueError(f"Cannot crop image shape {(h, w)} to {shape}")
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
    b, c, h, w, two = x.shape
    return x.permute(0, 1, 4, 2, 3).reshape(b, c * two, h, w)


def channels_to_complex_last(x: torch.Tensor, coils: int = 1) -> torch.Tensor:
    if x.ndim != 4 or x.shape[1] != coils * 2:
        raise ValueError(f"Expected channel-first tensor with {coils * 2} channels")
    b, _, h, w = x.shape
    return x.reshape(b, coils, 2, h, w).permute(0, 1, 3, 4, 2).contiguous()


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
