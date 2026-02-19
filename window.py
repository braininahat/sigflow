"""Main editor window wrapping NodeGraphQt graph widget."""
from __future__ import annotations

import logging
from pathlib import Path

import cv2

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtCore import QSize
from PySide6.QtWidgets import (
    QDockWidget, QMainWindow, QToolBar, QFileDialog, QLabel,
)

from NodeGraphQt import NodeGraph, PropertiesBinWidget, NodesTreeWidget
from NodeGraphQt.custom_widgets.nodes_tree import _BaseNodeTreeItem, TYPE_CATEGORY

from sigflow.registry import all_nodes
from sigflow_editor.nodes import register_visual_nodes
from sigflow_editor.bridge import EditorBridge

log = logging.getLogger(__name__)


def _restructure_palette(tree):
    """Restructure flat palette into Sources / Processing / Output hierarchy."""
    # Create three ordered parent groups
    parents = {}
    for label in ("Sources", "Processing", "Output"):
        item = _BaseNodeTreeItem(None, [label], type=TYPE_CATEGORY)
        item.setFirstColumnSpanned(True)
        item.setFlags(Qt.ItemIsEnabled)
        item.setSizeHint(0, QSize(100, 26))
        parents[label.lower()] = item

    # Map flat category keys to parent groups
    _GROUP_PARENT = {"source": "sources", "processing": "processing", "output": "output"}

    for cat_key, cat_item in list(tree._category_items.items()):
        if not cat_key.startswith("sigflow."):
            cat_item.setHidden(True)  # hide backdrop etc.
            continue

        parts = cat_key.split(".")  # e.g. ["sigflow", "source"] or ["sigflow", "processing", "transform"]
        group = parts[1]
        parent = parents.get(_GROUP_PARENT.get(group))
        if not parent:
            continue

        # Detach from top level
        idx = tree.indexOfTopLevelItem(cat_item)
        if idx >= 0:
            tree.takeTopLevelItem(idx)

        if len(parts) == 2:
            # No subcategory (source/output) — move node items directly under parent
            while cat_item.childCount() > 0:
                parent.addChild(cat_item.takeChild(0))
        else:
            # Has subcategory (processing.transform etc.) — nest as sub-group
            parent.addChild(cat_item)

    # Add parent groups in desired order
    for key in ("sources", "processing", "output"):
        tree.addTopLevelItem(parents[key])
        parents[key].setExpanded(True)


class EditorWindow(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("sigflow Pipeline Editor")
        self.resize(1200, 800)

        # Import built-in nodes into registry
        self._import_builtin_nodes()

        # NodeGraphQt setup
        self._graph = NodeGraph()
        register_visual_nodes(self._graph)

        # Bridge between editor and pipeline runtime
        self._bridge = EditorBridge(self._graph)

        # Live property editing → propagate to running pipeline
        self._graph.property_changed.connect(self._bridge.on_property_changed)

        # Right-click "Delete Selected" + Delete key shortcut
        graph_menu = self._graph.get_context_menu('graph')
        graph_menu.add_command(
            'Delete Selected',
            func=lambda graph: graph.delete_nodes(graph.selected_nodes()),
            shortcut='Delete',
        )

        # Layout
        graph_widget = self._graph.widget
        self.setCentralWidget(graph_widget)

        # Node palette (drag-and-drop to create nodes)
        nodes_tree = NodesTreeWidget(node_graph=self._graph)

        # Set subcategory labels (just the tag name — they'll nest under Processing)
        _KIND_GROUP = {"source": "source", "process": "processing", "sink": "output"}
        for spec in all_nodes().values():
            if spec.category:
                group = _KIND_GROUP[spec.kind]
                key = f"sigflow.{group}.{spec.category}"
                nodes_tree.set_category_label(key, spec.category.capitalize())

        # Restructure flat tree into Sources / Processing / Output hierarchy
        _restructure_palette(nodes_tree)

        nodes_dock = QDockWidget("Nodes")
        nodes_dock.setWidget(nodes_tree)
        self.addDockWidget(Qt.LeftDockWidgetArea, nodes_dock)

        # Properties panel (click a node to edit its properties here)
        properties_bin = PropertiesBinWidget(node_graph=self._graph)
        self._graph.node_double_clicked.disconnect(properties_bin.add_node)
        self._graph.node_selected.connect(properties_bin.add_node)
        properties_dock = QDockWidget("Properties")
        properties_dock.setWidget(properties_bin)
        self.addDockWidget(Qt.RightDockWidgetArea, properties_dock)

        # Toolbar
        toolbar = QToolBar("Pipeline")
        self.addToolBar(toolbar)

        toolbar.addAction("Start", self._on_start)
        toolbar.addAction("Stop", self._on_stop)
        toolbar.addSeparator()
        toolbar.addAction("Load YAML...", self._on_load)
        toolbar.addAction("Save YAML...", self._on_save)
        toolbar.addSeparator()
        toolbar.addAction("Auto Layout", self._on_auto_layout)
        toolbar.addSeparator()

        self._status_label = QLabel("Stopped")
        toolbar.addWidget(self._status_label)

        # Metrics polling timer
        self._metrics_timer = QTimer(self)
        self._metrics_timer.timeout.connect(self._update_metrics)
        self._metrics_timer.setInterval(200)

        # Display pump timer (GUI updates must run on main thread)
        self._display_timer = QTimer(self)
        self._display_timer.timeout.connect(self._pump_display)
        self._display_timer.setInterval(16)  # ~60fps

    def _import_builtin_nodes(self):
        import sigflow.nodes.webcam_source  # noqa: F401
        import sigflow.nodes.cv2_display  # noqa: F401
        import sigflow.nodes.crop  # noqa: F401
        import sigflow.nodes.audio_source  # noqa: F401
        import sigflow.nodes.spectrogram  # noqa: F401
        import sigflow.nodes.canvas_display  # noqa: F401
        import sigflow.nodes.dlc_inference  # noqa: F401
        import sigflow.nodes.keypoints_overlay  # noqa: F401
        import sigflow.nodes.face_mesh  # noqa: F401
        import sigflow.nodes.face_roi  # noqa: F401
        import sigflow.nodes.roi_crop  # noqa: F401
        import sigflow.nodes.mesh_overlay  # noqa: F401

    def _on_start(self):
        log.info("starting pipeline from editor")
        self._bridge.build_and_start()
        self._status_label.setText("Running")
        self._metrics_timer.start()
        self._display_timer.start()

    def _on_stop(self):
        log.info("stopping pipeline from editor")
        self._bridge.stop()
        self._status_label.setText("Stopped")
        self._metrics_timer.stop()
        self._display_timer.stop()

    def _on_load(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Pipeline", "", "YAML (*.yaml *.yml);;JSON (*.json)"
        )
        if path:
            log.info("loading graph: %s", path)
            self._bridge.load_graph(Path(path))
            self._graph.auto_layout_nodes()

    def _on_auto_layout(self):
        self._graph.auto_layout_nodes()

    def _on_save(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Pipeline", "", "YAML (*.yaml *.yml);;JSON (*.json)"
        )
        if path:
            log.info("saving graph: %s", path)
            self._bridge.save_graph(Path(path))

    def _pump_display(self):
        from sigflow.nodes.cv2_display import drain_display_queue
        drain_display_queue()

        from sigflow.nodes.canvas_display import _canvas_frames
        if not _canvas_frames:
            return
        for node in self._graph.all_nodes():
            if type(node)._REGISTRY_TYPE != "canvas_display":
                continue
            frame = _canvas_frames.get(self._bridge.node_clean_name(node))
            if frame is None:
                continue
            widget = node.view.get_widget("_preview")
            qlabel = widget.get_custom_widget()
            h, w = frame.shape[:2]
            if frame.ndim == 3:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888)
            else:
                qimg = QImage(frame.data, w, h, w, QImage.Format.Format_Grayscale8)
            pm = QPixmap.fromImage(qimg.copy())
            qlabel.setPixmap(pm.scaled(
                qlabel.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
            ))

    def _update_metrics(self):
        self._bridge.update_metrics_overlay()

    def closeEvent(self, event):
        log.info("editor window closing")
        self._display_timer.stop()
        self._bridge.stop()
        super().closeEvent(event)
