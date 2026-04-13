"""Graph definition and serialization (YAML/JSON) for sigflow pipelines."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path

import yaml

log = logging.getLogger(__name__)


@dataclass
class Connection:
    """A connection between two ports in a graph."""
    src_id: str
    src_port: str
    dst_id: str
    dst_port: str


@dataclass
class NodeDef:
    """A node instance definition in a graph."""
    id: str
    type: str
    config: dict = field(default_factory=dict)


@dataclass
class Graph:
    """Static DAG topology: nodes + connections + config. The blueprint."""
    nodes: list[NodeDef] = field(default_factory=list)
    connections: list[Connection] = field(default_factory=list)

    def _to_dict(self) -> dict:
        return {
            "nodes": [asdict(n) for n in self.nodes],
            "connections": [
                {"src": [c.src_id, c.src_port], "dst": [c.dst_id, c.dst_port]}
                for c in self.connections
            ],
        }

    @classmethod
    def _from_dict(cls, data: dict) -> Graph:
        nodes = [NodeDef(**n) for n in data.get("nodes", [])]
        connections = [
            Connection(
                src_id=c["src"][0], src_port=c["src"][1],
                dst_id=c["dst"][0], dst_port=c["dst"][1],
            )
            for c in data.get("connections", [])
        ]
        return cls(nodes=nodes, connections=connections)

    def save_yaml(self, path: Path) -> None:
        with open(path, "w") as f:
            yaml.dump(self._to_dict(), f, default_flow_style=False, sort_keys=False)
        log.info("saved graph to %s (%d nodes, %d connections)", path, len(self.nodes), len(self.connections))

    @classmethod
    def load_yaml(cls, path: Path) -> Graph:
        with open(path) as f:
            data = yaml.safe_load(f)
        graph = cls._from_dict(data)
        log.info("loaded graph from %s (%d nodes, %d connections)", path, len(graph.nodes), len(graph.connections))
        return graph

    def save_json(self, path: Path) -> None:
        with open(path, "w") as f:
            json.dump(self._to_dict(), f, indent=2)
        log.info("saved graph to %s (%d nodes, %d connections)", path, len(self.nodes), len(self.connections))

    @classmethod
    def load_json(cls, path: Path) -> Graph:
        with open(path) as f:
            data = json.load(f)
        graph = cls._from_dict(data)
        log.info("loaded graph from %s (%d nodes, %d connections)", path, len(graph.nodes), len(graph.connections))
        return graph
