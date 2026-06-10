from __future__ import annotations


DEFAULT_PROJECT_CONFIG = {
    "seed": 1337,
    "in_channels": 2,
    "out_channels": 2,
    "base_channels": 16,
    "channel_multipliers": (1, 2, 4),
    "conv_layers_per_block": 2,
    "activation": "silu",
    "normalization": "group",
    "upsample_mode": "transpose",
    "learning_rate": 1e-3,
}


def merge_project_config(config: dict | None) -> dict:
    merged = dict(DEFAULT_PROJECT_CONFIG)
    if config:
        merged.update(config)
    return merged
