"""
config_loader.py
================
Centralised configuration loader.  Reads config/config.yaml once and exposes
a frozen namespace so every module imports the same object without re-parsing.

Usage
-----
    from config.config_loader import CFG
    raw_path = CFG.paths.data_raw
    seed     = CFG.project.random_seed
"""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import yaml


def _to_namespace(d: Any) -> Any:
    """Recursively convert a dict to a SimpleNamespace for attribute access."""
    if isinstance(d, dict):
        return SimpleNamespace(**{k: _to_namespace(v) for k, v in d.items()})
    if isinstance(d, list):
        return [_to_namespace(i) for i in d]
    return d


def load_config(config_path: str | Path | None = None) -> SimpleNamespace:
    """
    Load the YAML configuration file and return it as a nested SimpleNamespace.

    Parameters
    ----------
    config_path : str or Path, optional
        Explicit path to config.yaml.  Defaults to ``<project_root>/config/config.yaml``,
        where project root is resolved as two levels above this file.
    """
    if config_path is None:
        # Resolve relative to this file: config_loader.py lives in config/
        config_path = Path(__file__).parent / "config.yaml"

    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {config_path}\n"
            "Make sure you are running from the project root."
        )

    with open(config_path, "r") as fh:
        raw: dict = yaml.safe_load(fh)

    return _to_namespace(raw)


# ---------------------------------------------------------------------------
# Module-level singleton — import CFG everywhere; never re-parse the YAML.
# ---------------------------------------------------------------------------
CFG: SimpleNamespace = load_config()

# Resolve the project root so other modules can construct absolute paths easily.
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]


def abs_path(relative: str) -> Path:
    """Return an absolute path given a path relative to the project root."""
    return PROJECT_ROOT / relative
