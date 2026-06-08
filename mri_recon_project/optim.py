from __future__ import annotations

import torch


def build_optimizer(parameters, learning_rate: float):
    return torch.optim.Adam(parameters, lr=learning_rate)

