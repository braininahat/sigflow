"""Bridge between NodeGraphQt graph editor and sigflow pipeline runtime."""
from __future__ import annotations

import logging
from pathlib import Path

from sigflow.graph import Graph, NodeDef, Connection
from sigflow.runtime import Pipeline

log = logging.getLogger(__name__)


class EditorBridge:
    """Syncs NodeGraphQt graph edits with sigflow Pipeline runtime."""

    def __init__(self, node_graph):
        self._node_graph = node_graph
        self._pipeline: Pipeline | None = None

    def _extract_graph(self) -> Graph:
        """Convert the current NodeGraphQt visual graph to a sigflow Graph."""
        nodes = []
        connections = []

        for node_id, node in self._node_graph.all_nodes().items():
            node_type = node.NODE_NAME
            config = {}
            for prop_name, prop_val in node.model.custom_properties.items():
                config[prop_name] = prop_val

            nodes.append(NodeDef(id=node.name(), type=node_type, config=config))

        for node_id, node in self._node_graph.all_nodes().items():
            for port in node.output_ports():
                for connected_port in port.connected_ports():
                    connections.append(Connection(
                        src_id=node.name(),
                        src_port=port.name(),
                        dst_id=connected_port.node().name(),
                        dst_port=connected_port.name(),
                    ))

        return Graph(nodes=nodes, connections=connections)

    def build_and_start(self) -> None:
        """Build a Pipeline from the current visual graph and start it."""
        if self._pipeline:
            self.stop()

        graph = self._extract_graph()
        self._pipeline = Pipeline.from_graph(graph)
        self._pipeline.start()
        log.info("Pipeline started from editor")

    def stop(self) -> None:
        if self._pipeline:
            self._pipeline.stop()
            self._pipeline = None
            log.info("Pipeline stopped from editor")

    def update_metrics_overlay(self) -> None:
        """Update visual node overlays with pipeline metrics."""
        if not self._pipeline:
            return

        snapshots = self._pipeline.metrics_snapshot()
        for node_id, node in self._node_graph.all_nodes().items():
            metrics = snapshots.get(node.name())
            if metrics:
                fps_str = f"{metrics.fps:.1f} fps"
                latency_str = f"{metrics.avg_process_ms:.1f}ms"
                queue_str = f"q:{metrics.queue_depth}"
                node.set_property("name", f"{node.NODE_NAME}\n{fps_str} | {latency_str} | {queue_str}")

    def load_graph(self, path: Path) -> None:
        """Load a graph from YAML/JSON and populate the visual editor."""
        if path.suffix in (".yaml", ".yml"):
            graph = Graph.load_yaml(path)
        else:
            graph = Graph.load_json(path)

        self._node_graph.clear_session()

        node_map = {}
        for i, node_def in enumerate(graph.nodes):
            visual_cls = f"sigflow.{node_def.type}"
            visual_node = self._node_graph.create_node(visual_cls, name=node_def.id)
            visual_node.set_pos(i * 250, 0)
            node_map[node_def.id] = visual_node

        for conn in graph.connections:
            src_node = node_map.get(conn.src_id)
            dst_node = node_map.get(conn.dst_id)
            if src_node and dst_node:
                src_port = src_node.get_output(conn.src_port)
                dst_port = dst_node.get_input(conn.dst_port)
                if src_port and dst_port:
                    src_port.connect_to(dst_port)

    def save_graph(self, path: Path) -> None:
        """Save the current visual graph to YAML/JSON."""
        graph = self._extract_graph()
        if path.suffix in (".yaml", ".yml"):
            graph.save_yaml(path)
        else:
            graph.save_json(path)
