"""Configuration loading with small, explicit validation helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG_PATH = Path("config.yaml")


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    """Load the project YAML configuration.

    The returned object is a regular dictionary so every threshold written to an artifact can
    be serialized without custom encoders.
    """

    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file does not exist: {config_path}")
    with config_path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Configuration root must be a mapping: {config_path}")

    required = {"project", "data", "preprocessing", "features", "rules", "postprocessing", "model"}
    missing = required.difference(config)
    if missing:
        raise ValueError(f"Configuration is missing sections: {sorted(missing)}")
    return config


def class_postprocessing_config(config: dict[str, Any], label: str) -> dict[str, float]:
    """Merge postprocessing defaults with per-class overrides."""

    section = config["postprocessing"]
    merged = dict(section["default"])
    merged.update(section.get(label, {}))
    return {key: float(value) for key, value in merged.items()}
