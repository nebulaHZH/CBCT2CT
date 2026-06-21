from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

import yaml


def load_config(path: str | Path, overrides: Mapping[str, str] | None = None) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if not isinstance(config, dict):
        raise ValueError(f"Config file must contain a mapping: {path}")

    merged = deepcopy(config)
    for key, value in (overrides or {}).items():
        _set_nested(merged, key.split("."), _parse_scalar(value))
    return merged


def _set_nested(config: dict[str, Any], keys: list[str], value: Any) -> None:
    cursor = config
    for key in keys[:-1]:
        child = cursor.setdefault(key, {})
        if not isinstance(child, dict):
            raise ValueError(f"Cannot set nested override through non-mapping key: {key}")
        cursor = child
    cursor[keys[-1]] = value


def _parse_scalar(value: str) -> Any:
    parsed = yaml.safe_load(value)
    return value if parsed is None and value.lower() != "null" else parsed
