"""Pipeline runtime: Qt-free DAG execution with threading, deque queues, and metrics."""
from __future__ import annotations

import enum
import logging
import threading
import time
from collections import deque

from sigflow.graph import Connection
from sigflow.metrics import MetricsCollector, MetricsTracker
from sigflow.node import NodeSpec
from sigflow.registry import get as registry_get
from sigflow.types import Sample, compatible

log = logging.getLogger(__name__)


class PipelineMode(enum.Enum):
    STOPPED = "stopped"
    LIVE = "live"
    DRAINING = "draining"


class IncompatiblePortError(Exception):
    pass


class MasterClock:
    """LSL-based master clock for pipeline-wide timestamps."""

    def __init__(self, time_fn=None):
        self._time_fn = time_fn  # injectable for testing
        self._start_time: float = 0.0
        self._started = False

    def _now(self) -> float:
        if self._time_fn:
            return self._time_fn()
        import pylsl
        return pylsl.local_clock()

    def start(self) -> None:
        self._start_time = self._now()
        self._started = True

    def lsl_now(self) -> float:
        return self._now()

    def session_time_ms(self) -> int:
        if not self._started:
            return 0
        return int((self._now() - self._start_time) * 1000)


_PARAM_COERCE = {"int": int, "float": float, "str": str, "bool": bool}


class NodeInstance:
    """Wraps a node function with threading, queues, and metrics."""

    def __init__(self, node_id: str, spec: NodeSpec, config: dict, clock: MasterClock, pipeline: Pipeline):
        self.node_id = node_id
        self._spec = spec
        # Merge param defaults with user-provided config
        merged = {p.name: p.default for p in spec.params}
        merged.update(config)
        # Coerce types (NodeGraphQt returns all widget values as strings)
        for p in spec.params:
            if p.name in merged:
                coerce = _PARAM_COERCE.get(p.type)
                if coerce and not isinstance(merged[p.name], coerce):
                    merged[p.name] = coerce(merged[p.name])
        merged["_node_id"] = node_id
        self._config = merged
        self._state: dict = {}
        self._clock = clock
        self._pipeline = pipeline
        self._queues: dict[str, deque] = {
            port.name: deque() for port in spec.inputs
        }
        self._metrics: MetricsTracker | None = None
        self._thread: threading.Thread | None = None
        self._running = False
        self._event = threading.Event()

    def on_input(self, port_name: str, sample: Sample) -> None:
        """Thread-safe: push sample into input deque and wake worker."""
        if port_name in self._queues:
            self._queues[port_name].append(sample)
            self._event.set()

    def _process_queues(self) -> None:
        """Process one item from the first non-empty queue."""
        for port_name, q in self._queues.items():
            if not q:
                continue
            if self._pipeline._mode == PipelineMode.LIVE:
                item = q.pop()
                skipped = len(q)
                q.clear()
                if self._metrics and skipped > 0:
                    self._metrics.record_skipped(skipped)
            else:
                item = q.popleft()
            self._invoke(item)
            return

    def _invoke(self, item: Sample) -> None:
        """Call the node function and dispatch outputs."""
        t0 = time.perf_counter()
        try:
            if self._spec.kind == "sink":
                self._spec.func(item, state=self._state, config=self._config)
            else:
                result = self._spec.func(item, state=self._state, config=self._config)
                if result:
                    self._pipeline._dispatch(self.node_id, result)
        except Exception:
            log.exception("node '%s' raised during invoke", self.node_id)
            return
        elapsed_ms = (time.perf_counter() - t0) * 1000
        if self._metrics:
            self._metrics.record(elapsed_ms)

    def _run_source_loop(self) -> None:
        """Blocking loop for source nodes."""
        log.debug("source loop started: %s", self.node_id)
        consecutive_errors = 0
        while self._running:
            t0 = time.perf_counter()
            try:
                result = self._spec.func(state=self._state, config=self._config, clock=self._clock)
            except Exception:
                consecutive_errors += 1
                if consecutive_errors >= 3:
                    log.error("source '%s' failed %d times, stopping", self.node_id, consecutive_errors)
                    break
                log.exception("source '%s' raised during capture", self.node_id)
                time.sleep(0.5)
                continue
            consecutive_errors = 0
            if result:
                elapsed_ms = (time.perf_counter() - t0) * 1000
                if self._metrics:
                    self._metrics.record(elapsed_ms)
                self._pipeline._dispatch(self.node_id, result)
        log.debug("source loop exited: %s", self.node_id)

    def _run_worker_loop(self) -> None:
        """Event-driven loop for process/sink nodes."""
        while self._running:
            self._event.wait(timeout=0.1)
            self._event.clear()
            if not self._running:
                break
            has_items = True
            while has_items and self._running:
                has_items = False
                for q in self._queues.values():
                    if q:
                        has_items = True
                        break
                if has_items:
                    self._process_queues()

    def start(self) -> None:
        self._running = True
        if self._spec.kind == "source":
            self._thread = threading.Thread(
                target=self._run_source_loop, daemon=True, name=f"sigflow-{self.node_id}"
            )
        else:
            self._thread = threading.Thread(
                target=self._run_worker_loop, daemon=True, name=f"sigflow-{self.node_id}"
            )
        self._thread.start()
        log.debug("started thread for %s node '%s'", self._spec.kind, self.node_id)

    def stop(self) -> None:
        self._running = False
        self._event.set()
        if self._thread:
            self._thread.join(timeout=3.0)
            if self._thread.is_alive():
                log.warning("node '%s' thread did not stop within 3s", self.node_id)

    def init(self) -> None:
        if self._spec.init_func:
            log.debug("init: %s", self.node_id)
            self._spec.init_func(self._state, self._config)

    def cleanup(self) -> None:
        if self._spec.cleanup_func:
            log.debug("cleanup: %s", self.node_id)
            self._spec.cleanup_func(self._state, self._config)

    def drain(self) -> None:
        """Process all remaining queued items in FIFO order."""
        for q in self._queues.values():
            while q:
                item = q.popleft()
                self._invoke(item)

    def update_config(self, key: str, value) -> None:
        """Hot-update a config value (thread-safe for simple types under GIL)."""
        param_type = next(
            (p.type for p in self._spec.params if p.name == key), None
        )
        coerce = _PARAM_COERCE.get(param_type)
        if coerce and not isinstance(value, coerce):
            value = coerce(value)
        self._config[key] = value

    def queue_depth(self) -> int:
        return sum(len(q) for q in self._queues.values())


class Pipeline:
    """Qt-free DAG execution engine."""

    def __init__(self, max_workers: int = 4):
        self._nodes: dict[str, NodeInstance] = {}
        self._connections: list[Connection] = []
        self._clock = MasterClock(time_fn=time.monotonic)
        self._metrics = MetricsCollector()
        self._mode = PipelineMode.STOPPED
        self._max_workers = max_workers

    def add_node(self, node_id: str, node_type: str, config: dict) -> None:
        spec = registry_get(node_type)
        instance = NodeInstance(node_id, spec, config, self._clock, self)
        instance._metrics = self._metrics.create_tracker(node_id)
        self._nodes[node_id] = instance
        log.info("added %s node '%s' (type=%s)", spec.kind, node_id, node_type)

    def connect(self, src_id: str, src_port: str, dst_id: str, dst_port: str) -> None:
        src_node = self._nodes[src_id]
        dst_node = self._nodes[dst_id]

        src_port_type = None
        for port in src_node._spec.outputs:
            if port.name == src_port:
                src_port_type = port.type
                break
        if src_port_type is None:
            raise ValueError(f"Node '{src_id}' has no output port '{src_port}'")

        dst_port_type = None
        for port in dst_node._spec.inputs:
            if port.name == dst_port:
                dst_port_type = port.type
                break
        if dst_port_type is None:
            raise ValueError(f"Node '{dst_id}' has no input port '{dst_port}'")

        if not compatible(src_port_type, dst_port_type):
            raise IncompatiblePortError(
                f"Cannot connect {src_port_type.__name__} to {dst_port_type.__name__}"
            )

        self._connections.append(Connection(
            src_id=src_id, src_port=src_port, dst_id=dst_id, dst_port=dst_port,
        ))
        log.info("connected %s.%s -> %s.%s", src_id, src_port, dst_id, dst_port)

    def _dispatch(self, src_id: str, outputs: dict[str, Sample]) -> None:
        """Route outputs from a node to all connected downstream nodes."""
        for port_name, sample in outputs.items():
            for conn in self._connections:
                if conn.src_id == src_id and conn.src_port == port_name:
                    copy = sample.replace(metadata=dict(sample.metadata))
                    self._nodes[conn.dst_id].on_input(conn.dst_port, copy)

    def start(self) -> None:
        self._clock.start()
        self._mode = PipelineMode.LIVE

        for node in self._nodes.values():
            node.init()

        for node in self._nodes.values():
            if node._spec.kind != "source":
                node.start()

        for node in self._nodes.values():
            if node._spec.kind == "source":
                node.start()

        log.info("pipeline started: %d nodes, %d connections",
                 len(self._nodes), len(self._connections))

    def stop(self) -> None:
        log.info("stopping pipeline ...")
        for node in self._nodes.values():
            if node._spec.kind == "source":
                node.stop()

        time.sleep(0.05)

        self._mode = PipelineMode.DRAINING
        for node in self._nodes.values():
            if node._spec.kind != "source":
                node.stop()
                node.drain()

        for node in self._nodes.values():
            node.cleanup()

        self._mode = PipelineMode.STOPPED
        log.info("pipeline stopped")

    def drain(self) -> None:
        self._mode = PipelineMode.DRAINING
        for node in self._nodes.values():
            node.drain()

    def update_node_config(self, node_id: str, key: str, value) -> None:
        """Hot-update a config value on a running node."""
        node = self._nodes.get(node_id)
        if node:
            node.update_config(key, value)

    def metrics_snapshot(self) -> dict:
        snapshots = {}
        for node_id, node in self._nodes.items():
            if node._metrics:
                snapshots[node_id] = node._metrics.snapshot(queue_depth=node.queue_depth())
        return snapshots

    @classmethod
    def from_graph(cls, graph, **kwargs) -> Pipeline:
        p = cls(**kwargs)
        for node_def in graph.nodes:
            p.add_node(node_def.id, node_def.type, node_def.config)
        for conn in graph.connections:
            p.connect(conn.src_id, conn.src_port, conn.dst_id, conn.dst_port)
        return p
