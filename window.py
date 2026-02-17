"""Main editor window wrapping NodeGraphQt graph widget."""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import (
    QMainWindow, QToolBar, QFileDialog, QLabel,
)

from NodeGraphQt import NodeGraph

from sigflow_editor.nodes import register_visual_nodes
from sigflow_editor.bridge import EditorBridge


class EditorWindow(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("sigflow Pipeline Editor")
        self.resize(1200, 800)

        # Import built-in nodes into registry
        self._import_builtin_nodes()

        # NodeGraphQt setup
        self._graph = NodeGraph()
        self._graph.set_context_menu_from_file(None)
        register_visual_nodes(self._graph)

        # Bridge between editor and pipeline runtime
        self._bridge = EditorBridge(self._graph)

        # Layout
        graph_widget = self._graph.widget
        self.setCentralWidget(graph_widget)

        # Toolbar
        toolbar = QToolBar("Pipeline")
        self.addToolBar(toolbar)

        toolbar.addAction("Start", self._on_start)
        toolbar.addAction("Stop", self._on_stop)
        toolbar.addSeparator()
        toolbar.addAction("Load YAML...", self._on_load)
        toolbar.addAction("Save YAML...", self._on_save)
        toolbar.addSeparator()

        self._status_label = QLabel("Stopped")
        toolbar.addWidget(self._status_label)

        # Metrics polling timer
        self._metrics_timer = QTimer(self)
        self._metrics_timer.timeout.connect(self._update_metrics)
        self._metrics_timer.setInterval(200)

    def _import_builtin_nodes(self):
        import sigflow.nodes.webcam_source  # noqa: F401
        import sigflow.nodes.cv2_display  # noqa: F401
        import sigflow.nodes.crop  # noqa: F401
        import sigflow.nodes.audio_source  # noqa: F401
        import sigflow.nodes.spectrogram  # noqa: F401
        import sigflow.nodes.plot_display  # noqa: F401

    def _on_start(self):
        self._bridge.build_and_start()
        self._status_label.setText("Running")
        self._metrics_timer.start()

    def _on_stop(self):
        self._bridge.stop()
        self._status_label.setText("Stopped")
        self._metrics_timer.stop()

    def _on_load(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Pipeline", "", "YAML (*.yaml *.yml);;JSON (*.json)"
        )
        if path:
            self._bridge.load_graph(Path(path))

    def _on_save(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Pipeline", "", "YAML (*.yaml *.yml);;JSON (*.json)"
        )
        if path:
            self._bridge.save_graph(Path(path))

    def _update_metrics(self):
        self._bridge.update_metrics_overlay()

    def closeEvent(self, event):
        self._bridge.stop()
        super().closeEvent(event)
