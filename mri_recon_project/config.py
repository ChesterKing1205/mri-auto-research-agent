from __future__ import annotations


DEFAULT_PROJECT_CONFIG = {
    "seed": 1337,
    "in_channels": 2,
    "out_channels": 2,
    "base_channels": 16,
    "unet_depth": 3,
    "channel_multiplier": 2,
    "conv_layers_per_block": 2,
    "activation": "leaky_relu",
    "normalization": "instance",
    "upsample_mode": "transpose",
    "learning_rate": 1e-3,
}


def merge_project_config(config: dict | None) -> dict:
    merged = dict(DEFAULT_PROJECT_CONFIG)
    if config:
        merged.update(config)
    return merged
