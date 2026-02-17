"""Node decorators and NodeSpec for sigflow DAG nodes."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from sigflow.types import Port


@dataclass
class NodeSpec:
    """Metadata describing a node type."""
    name: str
    kind: str  # "source", "process", "sink"
    inputs: list[Port]
    outputs: list[Port]
    category: str
    func: Callable
    init_func: Callable | None = None
    cleanup_func: Callable | None = None


def _make_decorator(kind: str, *, require_inputs: bool, require_outputs: bool):
    """Factory for source_node, process_node, sink_node decorators."""

    def decorator(
        name: str,
        inputs: list[Port] | None = None,
        outputs: list[Port] | None = None,
        category: str = "",
    ):
        inputs = inputs or []
        outputs = outputs or []

        if require_inputs and not inputs:
            raise ValueError(f"{kind}_node '{name}' must have at least one input port")
        if require_outputs and not outputs:
            raise ValueError(f"{kind}_node '{name}' must have at least one output port")

        def wrapper(func: Callable) -> Callable:
            spec = NodeSpec(
                name=name,
                kind=kind,
                inputs=list(inputs),
                outputs=list(outputs),
                category=category,
                func=func,
            )
            func.spec = spec

            def init_decorator(init_fn: Callable) -> Callable:
                spec.init_func = init_fn
                return init_fn

            def cleanup_decorator(cleanup_fn: Callable) -> Callable:
                spec.cleanup_func = cleanup_fn
                return cleanup_fn

            func.init = init_decorator
            func.cleanup = cleanup_decorator

            # Auto-register (import registry lazily to avoid circular imports)
            from sigflow.registry import register
            register(spec)

            return func

        return wrapper

    return decorator


source_node = _make_decorator("source", require_inputs=False, require_outputs=True)
process_node = _make_decorator("process", require_inputs=True, require_outputs=True)
sink_node = _make_decorator("sink", require_inputs=True, require_outputs=False)
