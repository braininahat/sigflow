"""Centralized path resolution for sigflow — works in dev and frozen (PyInstaller) mode.

In development:  project_root = the repo root (where pyproject.toml lives)
When frozen:     project_root = sys._MEIPASS (PyInstaller's temp extraction dir)

All resource lookups go through resolve_path() so that both modes find
weights/, config/, assets/, third_party/, etc. without hardcoded parents[N].
"""
from __future__ import annotations

import sys
from pathlib import Path


def _find_project_root() -> Path:
    """Return the project root directory."""
    if getattr(sys, "frozen", False):
        # PyInstaller sets sys._MEIPASS to the temp extraction dir
        return Path(sys._MEIPASS)
    # Development: walk up from this file (src/sigflow/paths.py) to repo root
    return Path(__file__).resolve().parents[2]


PROJECT_ROOT = _find_project_root()


def _find_data_dir() -> Path:
    """Writable persistent dir for user data (weights, recordings)."""
    if getattr(sys, "frozen", False):
        return Path.home() / ".ultraspeech"
    return _find_project_root()


DATA_DIR = _find_data_dir()


def resolve_path(relative: str) -> Path:
    """Resolve a project-relative path for bundled read-only resources."""
    return PROJECT_ROOT / relative


def resolve_data_path(relative: str) -> Path:
    """Resolve a path for writable user data (e.g. 'weights/model.onnx').

    In frozen mode, checks ~/.ultraspeech first (downloaded/writable),
    then falls back to _MEIPASS (bundled read-only).
    """
    data_path = DATA_DIR / relative
    if data_path.exists():
        return data_path
    # Fall back to bundled resources (frozen) or repo root (dev)
    return PROJECT_ROOT / relative
