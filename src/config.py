"""Lightweight config loader.

We use OmegaConf for ergonomic dotted-access, falling back to a plain
``PyYAML`` load if OmegaConf is not installed (keeps the repo runnable
on a stripped-down environment).
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

try:
    from omegaconf import DictConfig, OmegaConf

    _HAS_OMEGACONF = True
except Exception:
    _HAS_OMEGACONF = False
    DictConfig = dict  # type: ignore

import yaml


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "default.yaml"


def load_config(path: str | Path | None = None) -> Any:
    """Load the YAML config and return an OmegaConf object (or plain dict)."""
    cfg_path = Path(path) if path else DEFAULT_CONFIG_PATH
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")
    if _HAS_OMEGACONF:
        return OmegaConf.load(str(cfg_path))
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dirs(cfg: Any) -> None:
    """Create all directories referenced under ``cfg.paths``."""
    paths = cfg["paths"] if isinstance(cfg, dict) else cfg.paths
    for _, p in paths.items():
        # skip file paths (anything with a dot in the last segment)
        last = os.path.basename(str(p))
        if "." in last:
            continue
        Path(p).mkdir(parents=True, exist_ok=True)
