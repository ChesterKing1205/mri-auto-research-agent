from __future__ import annotations

import numpy as np
import torch
from skimage.metrics import structural_similarity


def _to_numpy(x: torch.Tensor) -> np.ndarray:
    return x.detach().float().cpu().numpy()


def psnr(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    mse = torch.mean((pred - target) ** 2).clamp_min(1e-12)
    data_range = (target.max() - target.min()).clamp_min(1e-6)
    return 20 * torch.log10(data_range) - 10 * torch.log10(mse)


def nmse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    numerator = torch.sum((pred - target) ** 2)
    denominator = torch.sum(target**2).clamp_min(1e-12)
    return numerator / denominator


def ssim(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_np = _to_numpy(pred)
    target_np = _to_numpy(target)
    scores: list[float] = []
    for pred_item, target_item in zip(pred_np, target_np):
        pred_img = np.squeeze(pred_item)
        target_img = np.squeeze(target_item)
        data_range = float(target_img.max() - target_img.min())
        if data_range <= 0:
            data_range = 1.0
        scores.append(float(structural_similarity(target_img, pred_img, data_range=data_range)))
    return torch.tensor(float(np.mean(scores)), dtype=torch.float32, device=pred.device)


def compute_metrics(pred: torch.Tensor, target: torch.Tensor) -> dict[str, torch.Tensor]:
    return {
        "psnr": psnr(pred, target),
        "ssim": ssim(pred, target),
        "nmse": nmse(pred, target),
    }

