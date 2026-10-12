from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    config = deepcopy(config)
    config.setdefault("name", path.stem)
    config.setdefault("encoder", {})
    config.setdefault("model", {})
    config.setdefault("training", {})
    config["config_path"] = str(path)
    return config


def config_name(config: dict[str, Any]) -> str:
    name = config.get("name")
    if not name:
        raise ValueError("Config is missing required field: name")
    return str(name)


def training_value(config: dict[str, Any], key: str, default: Any) -> Any:
    return config.get("training", {}).get(key, default)
