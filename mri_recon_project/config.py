from __future__ import annotations


DEFAULT_PROJECT_CONFIG = {
    "in_channels": 1,
    "base_channels": 16,
    "learning_rate": 1e-3,
}


def merge_project_config(config: dict | None) -> dict:
    merged = dict(DEFAULT_PROJECT_CONFIG)
    if config:
        merged.update(config)
    return merged
