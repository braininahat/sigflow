"""Bridge between NodeGraphQt graph editor and sigflow pipeline runtime."""
from __future__ import annotations

import logging
from pathlib import Path

from sigflow.graph import Graph, NodeDef, Connection
from sigflow.registry import get as registry_get
from sigflow.runtime import Pipeline

log = logging.getLogger(__name__)


class EditorBridge:
    """Syncs NodeGraphQt graph edits with sigflow Pipeline runtime."""

    def __init__(self, node_graph):
        self._node_graph = node_graph
        self._pipeline: Pipeline | None = None
        self._node_id_map: dict[int, str] = {}  # id(node) → clean name

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
        if self._pipeline:
            self._pipeline.stop()
            self._pipeline = None
            # Restore clean names (undo metrics overlay pollution)
            for node in self._node_graph.all_nodes():
                clean = self.node_clean_name(node)
                node.set_property("name", clean)
            self._node_id_map.clear()
            log.info("Pipeline stopped from editor")

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
        if not self._pipeline or prop_name.startswith("_") or prop_name == "name":
            return
        clean_name = self.node_clean_name(node)
        self._pipeline.update_node_config(clean_name, prop_name, prop_value)

    def load_graph(self, path: Path) -> None:
        """Load a graph from YAML/JSON and populate the visual editor."""
        log.info("loading graph from %s", path)
        if path.suffix in (".yaml", ".yml"):
            graph = Graph.load_yaml(path)
        else:
            graph = Graph.load_json(path)

        self._node_graph.clear_session()

        node_map = {}
        for i, node_def in enumerate(graph.nodes):
            spec = registry_get(node_def.type)
            group = {"source": "source", "process": "processing", "sink": "output"}[spec.kind]
            identifier = f"sigflow.{group}.{spec.category}" if spec.category else f"sigflow.{group}"
            visual_cls = f"{identifier}.Visual_{node_def.type}"
            visual_node = self._node_graph.create_node(visual_cls, name=node_def.id)
            visual_node.set_pos(i * 250, 0)
            # Restore config values to node properties
            for key, val in node_def.config.items():
                if visual_node.has_property(key):
                    visual_node.set_property(key, val)
            node_map[node_def.id] = visual_node
            log.debug("created visual node '%s' (type=%s)", node_def.id, node_def.type)

        for conn in graph.connections:
            src_node = node_map.get(conn.src_id)
            dst_node = node_map.get(conn.dst_id)
            if src_node and dst_node:
                src_port = src_node.get_output(conn.src_port)
                dst_port = dst_node.get_input(conn.dst_port)
                if src_port and dst_port:
                    src_port.connect_to(dst_port)
                    log.debug("connected %s.%s -> %s.%s", conn.src_id, conn.src_port, conn.dst_id, conn.dst_port)

        log.info("loaded %d nodes, %d connections into editor", len(node_map), len(graph.connections))

    def save_graph(self, path: Path) -> None:
        """Save the current visual graph to YAML/JSON."""
        log.info("saving graph to %s", path)
        graph = self._extract_graph()
        if path.suffix in (".yaml", ".yml"):
            graph.save_yaml(path)
        else:
            graph.save_json(path)
