"""Auto-generate NodeGraphQt visual node classes from the sigflow registry."""
from __future__ import annotations

from NodeGraphQt import BaseNode

from sigflow.registry import all_nodes
from sigflow.node import NodeSpec


# Color map by node kind
_KIND_COLORS = {
    "source": (50, 150, 50),    # green
    "process": (50, 100, 180),  # blue
    "sink": (180, 80, 50),      # red/orange
}


def _make_visual_node_class(spec: NodeSpec) -> type:
    """Dynamically create a NodeGraphQt BaseNode subclass from a NodeSpec."""

    class VisualNode(BaseNode):
        __identifier__ = "sigflow"
        NODE_NAME = spec.name

        def __init__(self):
            super().__init__()
            color = _KIND_COLORS.get(spec.kind, (100, 100, 100))
            self.set_color(*color)

            for port in spec.inputs:
                self.add_input(port.name)
            for port in spec.outputs:
                self.add_output(port.name)

    # Unique class name for NodeGraphQt registration
    VisualNode.__name__ = f"Visual_{spec.name}"
    VisualNode.__qualname__ = f"Visual_{spec.name}"
    return VisualNode


def register_visual_nodes(graph) -> None:
    """Register all sigflow node types as NodeGraphQt visual nodes."""
    for name, spec in all_nodes().items():
        cls = _make_visual_node_class(spec)
        graph.register_node(cls)
