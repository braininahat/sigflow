"""Bridge between NodeGraphQt graph editor and sigflow pipeline runtime."""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from sigflow.graph import Graph, NodeDef, Connection
from sigflow.registry import get as registry_get
from sigflow.runtime import Pipeline
from sigflow_editor.nodes import apply_recording_color
from sigflow.types import (
    TimeSeries2D, AudioSignal, Keypoints, FaceLandmarks,
    Scalar, Event, ROI,
)

log = logging.getLogger(__name__)


@dataclass
class _LiveStream:
    source_id: str
    port_type_name: str
    entries: list[tuple] = field(default_factory=list)
    is_video: bool = False
    node_id: str | None = None


def _extract_summary(sample) -> tuple:
    """Pure function: extract a compact summary tuple from a Sample.

    Runs in pool threads — must be fast, no locks.
    """
    pt = sample.port_type
    ts = sample.lsl_timestamp

    if issubclass(pt, TimeSeries2D):
        return (ts,)
    elif issubclass(pt, AudioSignal):
        data = sample.data
        return (ts, float(np.min(data)), float(np.max(data)))
    elif issubclass(pt, (Keypoints, FaceLandmarks)):
        data = sample.data.astype(np.float64)
        return (ts, float(np.sqrt(np.mean(data ** 2))))
    elif issubclass(pt, Scalar):
        return (ts, float(sample.data))
    elif issubclass(pt, Event):
        return (ts, str(sample.data))
    elif issubclass(pt, ROI):
        data = sample.data.astype(np.float64)
        return (ts, float(np.sqrt(np.mean(data ** 2))))
    return (ts,)


def _on_recorded_sample(sample, node_id, lock, streams, t0_holder):
    """Pool-thread callback: extract summary and append under lock."""
    entry = _extract_summary(sample)
    label = node_id or sample.source_id
    key = (label, sample.port_type.__name__)
    is_video = issubclass(sample.port_type, TimeSeries2D)
    with lock:
        if t0_holder[0] is None:
            t0_holder[0] = sample.lsl_timestamp
            log.info("live: first sample at t=%.3f", sample.lsl_timestamp)
        new_stream = key not in streams
        if new_stream:
            streams[key] = _LiveStream(
                source_id=sample.source_id,
                port_type_name=sample.port_type.__name__,
                is_video=is_video,
                node_id=node_id,
            )
        streams[key].entries.append(entry)
        count = len(streams[key].entries)
    if new_stream:
        log.info("live: new stream %s/%s (node=%s, video=%s)", sample.source_id,
                 sample.port_type.__name__, node_id, is_video)
    log.debug("live: %s/%s now has %d entries", label,
              sample.port_type.__name__, count)


class EditorBridge:
    """Syncs NodeGraphQt graph edits with sigflow Pipeline runtime."""

    def __init__(self, node_graph):
        self._node_graph = node_graph
        self._pipeline: Pipeline | None = None
        self._owns_pipeline = True
        self._node_id_map: dict[int, str] = {}  # id(node) → clean name
        # Live timeline accumulator (created fresh each recording)
        self._live_lock: threading.Lock | None = None
        self._live_streams: dict[tuple[str, str], _LiveStream] | None = None
        self._live_t0: list[float | None] | None = None

    def node_clean_name(self, node) -> str:
        """Return the original clean name for a node (before metrics pollution)."""
        return self._node_id_map.get(id(node), node.name())

    def _extract_graph(self) -> Graph:
        """Convert the current NodeGraphQt visual graph to a sigflow Graph."""
        nodes = []
        connections = []

        for node in self._node_graph.all_nodes():
            node_type = type(node)._REGISTRY_TYPE
            config = {}
            for prop_name, prop_val in node.model.custom_properties.items():
                if not prop_name.startswith("_"):
                    config[prop_name] = prop_val

            nodes.append(NodeDef(id=self.node_clean_name(node), type=node_type, config=config))

        for node in self._node_graph.all_nodes():
            for port in node.output_ports():
                for connected_port in port.connected_ports():
                    connections.append(Connection(
                        src_id=self.node_clean_name(node),
                        src_port=port.name(),
                        dst_id=self.node_clean_name(connected_port.node()),
                        dst_port=connected_port.name(),
                    ))

        log.debug("extracted graph: %d nodes, %d connections", len(nodes), len(connections))
        return Graph(nodes=nodes, connections=connections)

    def build_and_start(self) -> None:
        """Build a Pipeline from the current visual graph and start it."""
        if self._pipeline:
            self.stop()

        # Capture clean names before metrics overlay can pollute them
        self._node_id_map.clear()
        for node in self._node_graph.all_nodes():
            self._node_id_map[id(node)] = node.name()

        graph = self._extract_graph()
        self._pipeline = Pipeline.from_graph(graph)
        self._pipeline.start()
        log.info("Pipeline started from editor")

    def stop(self) -> None:
        if not self._pipeline:
            return
        if self._owns_pipeline:
            self._pipeline.stop()
            log.info("Pipeline stopped from editor")
        else:
            log.info("Pipeline detached from editor")
        self._pipeline = None
        for node in self._node_graph.all_nodes():
            clean = self.node_clean_name(node)
            node.set_property("name", clean)
        self._node_id_map.clear()
        self._owns_pipeline = True

    def attach_pipeline(self, pipeline: Pipeline) -> None:
        """Attach to an externally-owned pipeline (don't stop on close)."""
        self._pipeline = pipeline
        self._owns_pipeline = False
        self._node_id_map.clear()
        for node in self._node_graph.all_nodes():
            self._node_id_map[id(node)] = node.name()
        log.info("attached to external pipeline (%d nodes)", len(self._node_id_map))

    def update_metrics_overlay(self) -> None:
        """Update visual node overlays with pipeline metrics."""
        if not self._pipeline:
            return

        snapshots = self._pipeline.metrics_snapshot()
        for node in self._node_graph.all_nodes():
            clean = self.node_clean_name(node)
            metrics = snapshots.get(clean)
            if metrics:
                fps_str = f"{metrics.fps:.1f} fps"
                latency_str = f"{metrics.avg_process_ms:.1f}ms"
                queue_str = f"q:{metrics.queue_depth}"
                node.set_property("name", f"{clean}\n{fps_str} | {latency_str} | {queue_str}")

    def on_property_changed(self, node, prop_name, prop_value) -> None:
        """Propagate editor property changes to running pipeline."""
        if prop_name == "recording":
            spec = registry_get(type(node)._REGISTRY_TYPE)
            apply_recording_color(node, spec.kind, prop_value)
        if not self._pipeline or prop_name.startswith("_") or prop_name == "name":
            return
        clean_name = self.node_clean_name(node)
        log.info("property: %s.%s = %r", clean_name, prop_name, prop_value)
        try:
            self._pipeline.update_node_config(clean_name, prop_name, prop_value)
        except Exception:
            log.warning("property update failed: %s.%s = %r", clean_name, prop_name, prop_value, exc_info=True)

    def populate_graph(self, graph: Graph) -> None:
        """Populate the visual editor from a sigflow Graph object."""
        self._node_graph.clear_session()

        node_map = {}
        for i, node_def in enumerate(graph.nodes):
            spec = registry_get(node_def.type)
            group = {"source": "source", "process": "processing", "sink": "output"}[spec.kind]
            identifier = f"sigflow.{group}.{spec.category}" if spec.category else f"sigflow.{group}"
            visual_cls = f"{identifier}.Visual_{node_def.type}"
            visual_node = self._node_graph.create_node(visual_cls, name=node_def.id)
            visual_node.set_pos(i * 250, 0)
            # Restore config values to node properties (coerce types for Qt widgets)
            param_types = {p.name: p.type for p in spec.params}
            _coerce = {"int": int, "float": float, "str": str, "bool": bool}
            for key, val in node_def.config.items():
                if visual_node.has_property(key):
                    coerce = _coerce.get(param_types.get(key))
                    if coerce and not isinstance(val, coerce):
                        val = coerce(val)
                    visual_node.set_property(key, val)
            if visual_node.has_property("recording"):
                apply_recording_color(visual_node, spec.kind, visual_node.get_property("recording"))
            node_map[node_def.id] = visual_node
            log.debug("created visual node '%s' (type=%s)", node_def.id, node_def.type)

        for conn in graph.connections:
            src_node = node_map.get(conn.src_id)
            dst_node = node_map.get(conn.dst_id)
            if not src_node or not dst_node:
                missing = conn.src_id if not src_node else conn.dst_id
                log.warning("populate_graph: skipping connection, node '%s' not found", missing)
                continue
            src_port = src_node.get_output(conn.src_port)
            dst_port = dst_node.get_input(conn.dst_port)
            if not src_port or not dst_port:
                missing_port = conn.src_port if not src_port else conn.dst_port
                log.warning("populate_graph: port '%s' not found on connection %s→%s",
                            missing_port, conn.src_id, conn.dst_id)
                continue
            src_port.connect_to(dst_port)
            log.debug("connected %s.%s -> %s.%s", conn.src_id, conn.src_port, conn.dst_id, conn.dst_port)

        log.info("populated %d nodes, %d connections into editor", len(node_map), len(graph.connections))

    def load_graph(self, path: Path) -> None:
        """Load a graph from YAML/JSON and populate the visual editor."""
        log.info("loading graph from %s", path)
        if path.suffix in (".yaml", ".yml"):
            graph = Graph.load_yaml(path)
        else:
            graph = Graph.load_json(path)
        self.populate_graph(graph)

    def start_recording(self, output_dir="recordings", **kwargs) -> None:
        if not self._pipeline:
            log.warning("start_recording: no pipeline running")
            return
        # Create fresh accumulator state
        lock = threading.Lock()
        streams: dict[tuple[str, str], _LiveStream] = {}
        t0_holder: list[float | None] = [None]
        self._live_lock = lock
        self._live_streams = streams
        self._live_t0 = t0_holder

        def on_sample(sample, node_id):
            _on_recorded_sample(sample, node_id, lock, streams, t0_holder)

        try:
            self._pipeline.start_recording(output_dir, on_sample=on_sample, **kwargs)
        except Exception:
            log.error("start_recording failed", exc_info=True)
            return
        log.info("start_recording: accumulator created, on_sample callback wired")

    def stop_recording(self):
        """Stop recording, clear accumulator, return session dir Path or None."""
        streams = self._live_streams
        if streams:
            total = sum(len(s.entries) for s in streams.values())
            log.info("stop_recording: clearing accumulator (%d streams, %d total entries)",
                     len(streams), total)
        else:
            log.info("stop_recording: no accumulator (no live data collected)")
        self._live_lock = None
        self._live_streams = None
        self._live_t0 = None
        if self._pipeline:
            return self._pipeline.stop_recording()
        return None

    def take_live_snapshot(self):
        """Main-thread: shallow-copy live streams under lock.

        Returns (streams_copy, t0) or (None, None) if not recording.
        """
        lock = self._live_lock
        streams = self._live_streams
        t0_holder = self._live_t0
        if lock is None or streams is None or t0_holder is None:
            log.debug("snapshot: no accumulator (not recording)")
            return None, None
        with lock:
            copy = {k: _LiveStream(
                source_id=v.source_id,
                port_type_name=v.port_type_name,
                entries=list(v.entries),
                is_video=v.is_video,
                node_id=v.node_id,
            ) for k, v in streams.items()}
            t0 = t0_holder[0]
        total = sum(len(s.entries) for s in copy.values())
        log.debug("snapshot: %d streams, t0=%s, total_entries=%d",
                  len(copy), t0, total)
        return copy, t0

    def apply_changes(self, pipeline_bridge) -> None:
        """Extract editor graph, restart the external pipeline, re-attach."""
        graph = self._extract_graph()
        pipeline_bridge.restart_with_graph(graph)
        self._pipeline = pipeline_bridge.current_pipeline
        self._owns_pipeline = False
        log.info("applied changes: re-attached to new pipeline (%d nodes)", len(graph.nodes))

    def save_graph(self, path: Path) -> None:
        """Save the current visual graph to YAML/JSON."""
        log.info("saving graph to %s", path)
        graph = self._extract_graph()
        if path.suffix in (".yaml", ".yml"):
            graph.save_yaml(path)
        else:
            graph.save_json(path)
