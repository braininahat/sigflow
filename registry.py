"""Global node type registry with plugin directory scanning."""
from __future__ import annotations

import importlib.util
import logging
from pathlib import Path

from sigflow.node import NodeSpec

log = logging.getLogger(__name__)

_registry: dict[str, NodeSpec] = {}


def register(spec: NodeSpec) -> None:
    """Register a node type. Called automatically by decorators."""
    _registry[spec.name] = spec
    log.debug("registered %s node '%s' (%d params)", spec.kind, spec.name, len(spec.params))


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
    log.info("scanning plugins in %s", directory)
    count = 0
    for path in sorted(directory.glob("*.py")):
        if path.name.startswith("_"):
            continue
        log.debug("loading plugin %s", path.name)
        module_spec = importlib.util.spec_from_file_location(path.stem, path)
        module = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(module)
        count += 1
    log.info("loaded %d plugin(s)", count)
    return count
