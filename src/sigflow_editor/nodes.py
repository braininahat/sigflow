"""Auto-generate NodeGraphQt visual node classes from the sigflow registry."""
from __future__ import annotations

import logging

from Qt import QtCore, QtWidgets
from NodeGraphQt import BaseNode
from NodeGraphQt.constants import Z_VAL_NODE_WIDGET, NodePropWidgetEnum
from NodeGraphQt.widgets.node_widgets import NodeBaseWidget

from sigflow.registry import all_nodes
from sigflow.node import NodeSpec

log = logging.getLogger(__name__)


class NodeImageDisplay(NodeBaseWidget):
    """Embedded image preview widget for canvas_display nodes."""

    def __init__(self, parent=None, name='', label=''):
        super().__init__(parent, name, label)
        self.setZValue(Z_VAL_NODE_WIDGET + 1)
        lbl = QtWidgets.QLabel()
        lbl.setMinimumSize(160, 120)
        lbl.setAlignment(QtCore.Qt.AlignCenter)
        lbl.setText("No signal")
        lbl.setStyleSheet("background: #1a1a1a; color: #666;")
        self.set_custom_widget(lbl)

    @property
    def type_(self):
        return 'ImageDisplayWidget'

    def get_value(self):
        return ''

    def set_value(self, value):
        pass


# Color map by node kind
_KIND_COLORS = {
    "source": (80, 170, 90),    # fresh green (captures)
    "process": (124, 99, 225),  # primary purple (#7C63E1)
    "sink": (167, 139, 250),    # secondary purple (#A78BFA)
}

_KIND_COLORS_REC = {
    "source": (110, 210, 120),  # bright green
    "process": (160, 140, 250), # bright purple
}


def apply_recording_color(node, spec_kind, recording):
    """Set node color based on recording status."""
    if recording and spec_kind in _KIND_COLORS_REC:
        node.set_color(*_KIND_COLORS_REC[spec_kind])
    else:
        node.set_color(*_KIND_COLORS.get(spec_kind, (100, 100, 100)))
    log.debug("recording color: %s → %s", node.name(), "recording" if recording else "normal")


_TYPE_COERCE = {"int": int, "float": float, "str": str, "bool": bool}

_ACRONYMS = {'dlc', 'cv2', 'roi'}

# Map node kind to palette group name
_KIND_GROUP = {"source": "source", "process": "processing", "sink": "output"}


def _display_name(name: str) -> str:
    """Convert registry name to friendly display name.

    'dlc_inference' → 'DLC Inference', 'roi_crop' → 'ROI Crop'
    """
    return ' '.join(
        p.upper() if p in _ACRONYMS else p.capitalize()
        for p in name.split('_')
    )


def _make_visual_node_class(spec: NodeSpec) -> type:
    """Dynamically create a NodeGraphQt BaseNode subclass from a NodeSpec."""

    # Capture spec in closure for __init__
    _spec = spec

    _group = _KIND_GROUP[_spec.kind]

    # Type coercion map: NodeGraphQt spinbox get_value() returns str,
    # which crashes QSpinBox.setValue() on undo/redo round-trips.
    _param_coerce = {
        p.name: _TYPE_COERCE[p.type]
        for p in _spec.params
        if p.type in _TYPE_COERCE
    }

    class VisualNode(BaseNode):
        __identifier__ = f"sigflow.{_group}.{_spec.category}" if _spec.category else f"sigflow.{_group}"
        NODE_NAME = _display_name(_spec.name)
        _REGISTRY_TYPE = _spec.name

        def set_property(self, name, value, push_undo=True):
            coerce = _param_coerce.get(name)
            if coerce and not isinstance(value, coerce):
                value = coerce(value)
            super().set_property(name, value, push_undo)

        def __init__(self):
            super().__init__()
            color = _KIND_COLORS.get(_spec.kind, (100, 100, 100))
            self.set_color(*color)

            for port in _spec.inputs:
                self.add_input(port.name)
            for port in _spec.outputs:
                self.add_output(port.name)

            # Register properties and create inline widgets (view-only)
            for param in _spec.params:
                label = param.label or param.name
                if param.type == "int":
                    self.add_spinbox(
                        param.name, label=label,
                        value=param.default,
                        min_value=int(param.min) if param.min is not None else 0,
                        max_value=int(param.max) if param.max is not None else 10000,
                    )
                elif param.type == "float":
                    self.add_spinbox(
                        param.name, label=label,
                        value=param.default,
                        min_value=float(param.min) if param.min is not None else 0.0,
                        max_value=float(param.max) if param.max is not None else 1.0,
                        double=True,
                    )
                elif param.type == "str":
                    self.add_text_input(param.name, label=label, text=str(param.default))
                elif param.type == "bool":
                    self.add_checkbox(param.name, label=label, state=param.default)
                elif param.type == "choice":
                    self.add_combo_menu(param.name, label=label, items=param.choices or [])

            # Add recording property to non-sink nodes (properties-panel only)
            if _spec.kind != "sink":
                self.create_property(
                    "recording", value=False,
                    widget_type=NodePropWidgetEnum.QCHECK_BOX.value,
                    widget_tooltip="Record outputs from this node",
                )

            # Embed image preview for canvas_display nodes
            if _spec.name == "canvas_display":
                preview = NodeImageDisplay(self.view, name="_preview", label="")
                self.add_custom_widget(preview)

    # Unique class name for NodeGraphQt registration
    VisualNode.__name__ = f"Visual_{spec.name}"
    VisualNode.__qualname__ = f"Visual_{spec.name}"
    return VisualNode


def register_visual_nodes(graph) -> None:
    """Register all sigflow node types as NodeGraphQt visual nodes."""
    for name, spec in all_nodes().items():
        cls = _make_visual_node_class(spec)
        graph.register_node(cls)
        log.debug("registered visual node '%s' (%d params)", name, len(spec.params))
    log.info("registered %d visual node types", len(all_nodes()))
