"""Global node type registry with plugin directory scanning."""
from __future__ import annotations

import importlib.util
from pathlib import Path

from sigflow.node import NodeSpec


_registry: dict[str, NodeSpec] = {}


def register(spec: NodeSpec) -> None:
    """Register a node type. Called automatically by decorators."""
    _registry[spec.name] = spec


def get(name: str) -> NodeSpec:
    """Get a registered node type by name. Raises KeyError if not found."""
    return _registry[name]


def all_nodes() -> dict[str, NodeSpec]:
    """Return a copy of the full registry."""
    return dict(_registry)


def clear() -> None:
    """Clear the registry. Used in tests."""
    _registry.clear()


def scan_plugins(directory: Path) -> int:
    """Import all .py files from directory. Decorators auto-register. Returns count loaded."""
    count = 0
    for path in sorted(directory.glob("*.py")):
        if path.name.startswith("_"):
            continue
        module_spec = importlib.util.spec_from_file_location(path.stem, path)
        module = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(module)
        count += 1
    return count
