"""Main editor window wrapping NodeGraphQt graph widget."""
from __future__ import annotations

import logging
from pathlib import Path

import cv2

from PySide6.QtCore import QPointF, QRectF, Qt, QTimer, QSize
from PySide6.QtGui import QColor, QIcon, QImage, QPainter, QPen, QPixmap, QPolygonF
from PySide6.QtWidgets import (
    QDockWidget, QMainWindow, QToolBar, QFileDialog, QLabel,
)


def _make_icon(draw_fn, size=16):
    """Create a QIcon by painting on a transparent pixmap."""
    pm = QPixmap(size, size)
    pm.fill(Qt.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.Antialiasing)
    draw_fn(p, size)
    p.end()
    return QIcon(pm)


def _icon_play(p: QPainter, s: int):
    p.setBrush(QColor(80, 180, 80))
    p.setPen(Qt.NoPen)
    p.drawPolygon(QPolygonF([QPointF(3, 2), QPointF(3, s - 2), QPointF(s - 2, s / 2)]))


def _icon_stop(p: QPainter, s: int):
    p.setBrush(QColor(200, 70, 70))
    p.setPen(Qt.NoPen)
    p.drawRect(3, 3, s - 6, s - 6)


def _icon_refresh(p: QPainter, s: int):
    p.setPen(QPen(QColor(70, 130, 220), 2))
    p.drawArc(QRectF(2, 2, s - 4, s - 4), 30 * 16, 300 * 16)
    p.setBrush(QColor(70, 130, 220))
    p.setPen(Qt.NoPen)
    p.drawPolygon(QPolygonF([QPointF(s - 3, 1), QPointF(s - 3, 7), QPointF(s + 1, 4)]))


def _icon_record(p: QPainter, s: int):
    p.setBrush(QColor(220, 50, 50))
    p.setPen(Qt.NoPen)
    p.drawEllipse(3, 3, s - 6, s - 6)


def _icon_stop_rec(p: QPainter, s: int):
    p.setBrush(QColor(80, 80, 80))
    p.setPen(Qt.NoPen)
    p.drawRect(3, 3, s - 6, s - 6)


def _icon_folder(p: QPainter, s: int):
    p.setPen(Qt.NoPen)
    p.setBrush(QColor(200, 170, 60))
    p.drawRoundedRect(1, 4, s - 2, s - 5, 2, 2)
    p.drawRoundedRect(1, 3, s // 2, 3, 1, 1)


def _icon_save(p: QPainter, s: int):
    p.setPen(Qt.NoPen)
    p.setBrush(QColor(100, 140, 200))
    p.drawRoundedRect(2, 2, s - 4, s - 4, 2, 2)
    p.setBrush(QColor(255, 255, 255))
    p.drawRect(4, 2, s - 8, 5)
    p.drawRect(4, s - 6, s - 8, 4)


def _icon_layout(p: QPainter, s: int):
    p.setPen(QPen(QColor(120, 100, 200), 1.5))
    p.setBrush(Qt.NoBrush)
    p.drawRect(2, 2, 5, 5)
    p.drawRect(9, 2, 5, 5)
    p.drawRect(2, 9, 5, 5)
    p.drawRect(9, 9, 5, 5)


def _icon_check(p: QPainter, s: int):
    p.setPen(QPen(QColor(80, 180, 80), 2.5))
    p.drawLine(QPointF(3, s / 2), QPointF(6, s - 3))
    p.drawLine(QPointF(6, s - 3), QPointF(s - 2, 3))


from NodeGraphQt import NodeGraph, PropertiesBinWidget, NodesTreeWidget
from NodeGraphQt.custom_widgets.nodes_tree import _BaseNodeTreeItem, TYPE_CATEGORY

from sigflow.graph import Graph
from sigflow.runtime import Pipeline
from sigflow.reader import SessionReader
from sigflow.registry import all_nodes
from sigflow_editor.nodes import register_visual_nodes
from sigflow_editor.bridge import EditorBridge
from sigflow_editor.timeline import TimelinePanel

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
    def __init__(self, graph: Graph | None = None, pipeline: Pipeline | None = None,
                 pipeline_bridge=None, protocol_service=None, protocol_name: str | None = None,
                 parent=None):
        super().__init__(parent)
        log.info("EditorWindow.__init__ begin")
        self.setWindowTitle("sigflow Pipeline Editor")
        self.resize(1200, 800)
        self._pipeline_bridge = pipeline_bridge
        self._protocol_service = protocol_service
        self._protocol_name = protocol_name

        # Import built-in nodes into registry
        log.info("importing built-in node modules...")
        self._import_builtin_nodes()
        log.info("built-in node modules imported (%d nodes registered)", len(all_nodes()))

        # NodeGraphQt setup
        log.info("creating NodeGraph...")
        self._graph = NodeGraph()
        log.info("registering visual nodes...")
        register_visual_nodes(self._graph)
        log.info("visual nodes registered")

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
        log.info("setting central widget (NodeGraph canvas)...")
        graph_widget = self._graph.widget
        self.setCentralWidget(graph_widget)

        # Node palette (drag-and-drop to create nodes)
        log.info("building node palette...")
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
        log.info("palette built (%d top-level groups)", nodes_tree.topLevelItemCount())

        nodes_dock = QDockWidget("Nodes")
        nodes_dock.setWidget(nodes_tree)
        self.addDockWidget(Qt.LeftDockWidgetArea, nodes_dock)

        # Properties panel (click a node to edit its properties here)
        log.info("building properties panel...")
        properties_bin = PropertiesBinWidget(node_graph=self._graph)
        self._graph.node_double_clicked.disconnect(properties_bin.add_node)
        self._graph.node_selected.connect(properties_bin.add_node)
        properties_dock = QDockWidget("Properties")
        properties_dock.setWidget(properties_bin)
        self.addDockWidget(Qt.RightDockWidgetArea, properties_dock)

        # Timeline panel (bottom dock)
        log.info("building timeline panel...")
        self._timeline_panel = TimelinePanel()
        timeline_dock = QDockWidget("Timeline")
        timeline_dock.setWidget(self._timeline_panel)
        self.addDockWidget(Qt.BottomDockWidgetArea, timeline_dock)

        # Stylesheet (light chrome — NodeGraphQt canvas stays dark)
        self.setStyleSheet("""
            QMainWindow { background: #F9F8FC; }
            QToolBar { background: #FFFFFF; border-bottom: 1px solid #E2DFED; spacing: 4px; padding: 4px; }
            QToolBar QToolButton { padding: 4px 8px; border-radius: 4px; }
            QToolBar QToolButton:hover { background: #F1EFF7; }
            QDockWidget { font-weight: 500; }
            QDockWidget::title { background: #F1EFF7; padding: 6px; }
            QLabel#status { color: #534D64; font-size: 12px; }
        """)

        # Toolbar
        log.info("building toolbar...")
        toolbar = QToolBar("Pipeline")
        toolbar.setIconSize(QSize(16, 16))
        self.addToolBar(toolbar)

        self._start_action = toolbar.addAction(_make_icon(_icon_play), "Start", self._on_start)
        self._stop_action = toolbar.addAction(_make_icon(_icon_stop), "Stop", self._on_stop)
        self._apply_action = toolbar.addAction(_make_icon(_icon_refresh), "Apply Changes", self._on_apply_changes)
        toolbar.addSeparator()
        self._record_action = toolbar.addAction(_make_icon(_icon_record), "Record", self._on_record)
        self._stop_record_action = toolbar.addAction(_make_icon(_icon_stop_rec), "Stop Recording", self._on_stop_record)
        toolbar.addSeparator()
        toolbar.addAction(_make_icon(_icon_folder), "Load YAML...", self._on_load)
        toolbar.addAction(_make_icon(_icon_save), "Save YAML...", self._on_save)
        toolbar.addSeparator()
        toolbar.addAction(_make_icon(_icon_layout), "Auto Layout", self._on_auto_layout)
        toolbar.addSeparator()
        toolbar.addAction(_make_icon(_icon_folder), "Open Session...", self._on_open_session)
        toolbar.addSeparator()
        self._commit_action = toolbar.addAction(
            _make_icon(_icon_check), "Save Pipeline", self._on_commit_protocol,
        )
        self._commit_action.setEnabled(
            self._protocol_service is not None and self._protocol_name is not None
        )
        toolbar.addSeparator()

        self._status_label = QLabel("Stopped")
        self._status_label.setObjectName("status")
        toolbar.addWidget(self._status_label)

        # Metrics polling timer
        self._metrics_timer = QTimer(self)
        self._metrics_timer.timeout.connect(self._update_metrics)
        self._metrics_timer.setInterval(200)

        # Live timeline timer (polls bridge for live recording data)
        self._live_timeline_timer = QTimer(self)
        self._live_timeline_timer.timeout.connect(self._update_live_timeline)
        self._live_timeline_timer.setInterval(500)

        # Display pump timer (GUI updates must run on main thread)
        self._display_timer = QTimer(self)
        self._display_timer.timeout.connect(self._pump_display)
        self._display_timer.setInterval(16)  # ~60fps

        # Pre-populate editor with provided graph (e.g. from running pipeline)
        self._attached = pipeline is not None
        if graph:
            self._bridge.populate_graph(graph)
            self._graph.auto_layout_nodes()
            if pipeline:
                self._bridge.attach_pipeline(pipeline)
                self._metrics_timer.start()
                self._display_timer.start()
                self._status_label.setText("Attached")
        if self._attached:
            self._start_action.setEnabled(False)
            self._stop_action.setEnabled(False)
            self._record_action.setEnabled(False)
            self._stop_record_action.setEnabled(False)
            self._apply_action.setEnabled(self._pipeline_bridge is not None)
            log.info("editor opened in attached mode — Start/Stop/Record disabled")
        else:
            self._apply_action.setEnabled(False)
        log.info("EditorWindow.__init__ complete")

    def _import_builtin_nodes(self):
        _modules = [
            "sigflow.nodes.webcam_source",
            "sigflow.nodes.cv2_display",
            "sigflow.nodes.crop",
            "sigflow.nodes.audio_source",
            "sigflow.nodes.spectrogram",
            "sigflow.nodes.canvas_display",
            "sigflow.nodes.dlc_inference",
            "sigflow.nodes.keypoints_overlay",
            "sigflow.nodes.face_mesh",
            "sigflow.nodes.face_roi",
            "sigflow.nodes.roi_crop",
            "sigflow.nodes.mesh_overlay",
            "sigflow.nodes.scrcpy_screen",
            "sigflow.nodes.scrcpy_camera",
            "sigflow.nodes.scrcpy_mic",
            "sigflow.nodes.sonostar_source",
        ]
        import importlib
        import time as _time
        for mod_name in _modules:
            log.info("  importing %s ...", mod_name)
            t0 = _time.perf_counter()
            try:
                importlib.import_module(mod_name)
                log.info("  imported  %s (%.2fs)", mod_name, _time.perf_counter() - t0)
            except Exception:
                log.warning("  FAILED   %s (%.2fs)", mod_name, _time.perf_counter() - t0, exc_info=True)

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

    def _on_record(self):
        log.info("record button: pipeline_running=%s",
                 self._bridge._pipeline is not None)
        self._bridge.start_recording()
        self._status_label.setText("Recording")
        self._live_timeline_timer.start()

    def _on_stop_record(self):
        self._live_timeline_timer.stop()
        session_dir = self._bridge.stop_recording()
        log.info("stop-record: session_dir=%s", session_dir)
        self._status_label.setText("Running")
        if session_dir:
            reader = SessionReader(session_dir)
            self._timeline_panel.load_session(reader)

    def _on_apply_changes(self):
        log.info("applying changes from editor")
        self._bridge.apply_changes(self._pipeline_bridge)
        self._status_label.setText("Attached")

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

    def _on_open_session(self):
        dir_path = QFileDialog.getExistingDirectory(self, "Open Session")
        if dir_path:
            log.info("opening session: %s", dir_path)
            reader = SessionReader(Path(dir_path))
            self._timeline_panel.load_session(reader)

    def _on_commit_protocol(self):
        log.info("saving pipeline to protocol file '%s'", self._protocol_name)
        self._bridge.commit_to_protocol(self._protocol_service, self._protocol_name)
        self._status_label.setText("Pipeline Saved")

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
        delivered = 0
        for node in self._graph.all_nodes():
            if type(node)._REGISTRY_TYPE != "canvas_display":
                continue
            frame = _canvas_frames.get(self._bridge.node_clean_name(node))
            if frame is None:
                continue
            try:
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
                delivered += 1
            except Exception:
                log.warning("display pump failed for node %s", self._bridge.node_clean_name(node), exc_info=True)
        self._pump_count = getattr(self, "_pump_count", 0) + 1
        if delivered and self._pump_count % 60 == 0:
            log.debug("display pump: delivered %d frames (tick %d)", delivered, self._pump_count)

    def _update_metrics(self):
        self._bridge.update_metrics_overlay()
        self._metrics_tick = getattr(self, "_metrics_tick", 0) + 1
        if self._metrics_tick % 25 == 0:  # every ~5s at 200ms interval
            snapshot = self._bridge._pipeline.metrics_snapshot() if self._bridge._pipeline else {}
            if snapshot:
                summary = ", ".join(f"{k}:{v.fps:.0f}fps" for k, v in snapshot.items())
                log.debug("metrics: %s", summary)

    def _update_live_timeline(self):
        snapshot, t0 = self._bridge.take_live_snapshot()
        if snapshot is None or t0 is None:
            log.debug("live timeline tick: no snapshot (not recording?)")
            return
        from sigflow_editor.timeline import build_live_tracks
        tracks, t1 = build_live_tracks(snapshot, t0)
        log.debug("live timeline tick: %d streams, %d tracks, t0=%.3f t1=%.3f",
                  len(snapshot), len(tracks), t0, t1)
        self._timeline_panel.set_live_tracks(tracks, t0, t1)

    def closeEvent(self, event):
        log.info("editor window closing")
        self._display_timer.stop()
        self._metrics_timer.stop()
        self._bridge.stop()
        super().closeEvent(event)
