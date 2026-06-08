from __future__ import annotations

import torch
import torch.nn.functional as F


def reconstruction_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.l1_loss(pred, target)

