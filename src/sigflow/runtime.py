"""Pipeline runtime: reactive DAG execution with generation-based correlation."""
from __future__ import annotations

import enum
import logging
import os
import queue
import threading
import time
from collections import deque
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor

from sigflow.graph import Connection
from sigflow.metrics import MetricsCollector, MetricsTracker
from sigflow.node import NodeSpec
from sigflow.recorder import SessionRecorder
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


# ---------------------------------------------------------------------------
# NodeInstance
# ---------------------------------------------------------------------------

_PARAM_COERCE = {"int": int, "float": float, "str": str, "bool": bool}


class NodeInstance:
    """Wraps a node function with generation-based pending map and metrics."""

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
        self._lock = threading.Lock()
        self._inbox: deque[tuple[str, Sample]] = deque()
        self._metrics: MetricsTracker | None = None
        self._thread: threading.Thread | None = None
        self._running = False
        self._connected_ports: set[str] = set()
        self._output_queue: queue.Queue | None = None
        self._dispatch_thread: threading.Thread | None = None
        self._failed = False
        self._consecutive_errors = 0
        self._last_drop_warn_t: float = 0.0

    # Cap the per-node inbox.  Arrivals are serialized FIFO and processed
    # one at a time on the pool thread; if the body is slower than the
    # incoming rate, the oldest pending arrivals are dropped and a
    # throttled warning is logged.
    INBOX_CAP = 60

    def on_input(self, port_name: str, sample: Sample) -> None:
        """Thread-safe: append (port, sample) to inbox and wake the scheduler."""
        if self._failed:
            return
        dropped = False
        with self._lock:
            if len(self._inbox) >= self.INBOX_CAP:
                self._inbox.popleft()
                dropped = True
            self._inbox.append((port_name, sample))
            depth = len(self._inbox)
        log.debug("on_input: %s.%s fid=%d (inbox=%d)", self.node_id, port_name, sample.frame_id, depth)
        if dropped:
            now = time.monotonic()
            if now - self._last_drop_warn_t >= 1.0:
                log.warning("node '%s' inbox overflow > %d — dropped oldest", self.node_id, self.INBOX_CAP)
                self._last_drop_warn_t = now
        self._pipeline._schedule_node(self)

    def _invoke(self, port_name: str, sample: Sample) -> None:
        """Call the node function with a single (port, sample) and dispatch outputs."""
        if self._failed:
            return
        log.debug("invoke: %s.%s fid=%d", self.node_id, port_name, sample.frame_id)
        t0 = time.perf_counter()
        try:
            if self._spec.kind == "sink":
                self._spec.func(sample, state=self._state, config=self._config)
            else:
                result = self._spec.func(sample, state=self._state, config=self._config)
                if result:
                    self._pipeline._dispatch(self.node_id, result)
            self._consecutive_errors = 0
        except Exception:
            self._consecutive_errors += 1
            if self._consecutive_errors == 3:
                log.error("node '%s' failed %d consecutive times, disabling",
                          self.node_id, self._consecutive_errors)
                self._failed = True
            else:
                log.exception("node '%s' raised during invoke", self.node_id)
            return
        elapsed_ms = (time.perf_counter() - t0) * 1000
        if self._metrics:
            self._metrics.record(elapsed_ms)

    def _run_source_loop(self) -> None:
        """Blocking loop for source nodes. Stamps monotonic frame_id on all outputs."""
        log.debug("source loop started: %s", self.node_id)
        try:
            os.nice(-5)
            log.debug("source thread '%s' priority elevated (nice -5)", self.node_id)
        except (PermissionError, OSError, AttributeError):
            pass
        consecutive_errors = 0
        while self._running:
            t0 = time.perf_counter()
            try:
                result = self._spec.func(state=self._state, config=self._config, clock=self._clock)
                consecutive_errors = 0
                if result:
                    fid = self._state.get("_frame_counter", 0)
                    self._state["_frame_counter"] = fid + 1
                    for sample in result.values():
                        sample.frame_id = fid
                    elapsed_ms = (time.perf_counter() - t0) * 1000
                    if self._metrics:
                        self._metrics.record(elapsed_ms)
                    self._output_queue.put(result)
            except Exception:
                consecutive_errors += 1
                if consecutive_errors >= 3:
                    log.error("source '%s' failed %d times, stopping", self.node_id, consecutive_errors)
                    break
                log.exception("source '%s' raised", self.node_id)
                time.sleep(0.5)
        log.debug("source loop exited: %s", self.node_id)

    def _dispatch_loop(self) -> None:
        """Drain output queue and dispatch to downstream nodes. Dedicated thread per source."""
        while True:
            try:
                item = self._output_queue.get(timeout=0.1)
                if item is None:
                    break  # sentinel from stop()
                self._pipeline._dispatch(self.node_id, item)
            except queue.Empty:
                if not self._running:
                    break

    def start(self) -> None:
        """Start source node thread. Non-source nodes don't get threads."""
        self._running = True
        if self._spec.kind == "source":
            self._output_queue = queue.Queue()
            self._dispatch_thread = threading.Thread(
                target=self._dispatch_loop, daemon=True,
                name=f"sigflow-dispatch-{self.node_id}",
            )
            self._dispatch_thread.start()
            self._thread = threading.Thread(
                target=self._run_source_loop, daemon=True, name=f"sigflow-{self.node_id}"
            )
            self._thread.start()
            log.debug("started source thread for '%s'", self.node_id)

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=3.0)
            if self._thread.is_alive():
                log.warning("node '%s' source thread did not stop within 3s", self.node_id)
        if self._output_queue is not None:
            self._output_queue.put(None)  # sentinel
        if self._dispatch_thread:
            self._dispatch_thread.join(timeout=3.0)
            if self._dispatch_thread.is_alive():
                log.warning("node '%s' dispatch thread did not stop within 3s", self.node_id)

    def init(self) -> None:
        if self._spec.init_func:
            log.debug("init: %s", self.node_id)
            self._spec.init_func(self._state, self._config)

    def cleanup(self) -> None:
        if self._spec.cleanup_func:
            log.debug("cleanup: %s", self.node_id)
            self._spec.cleanup_func(self._state, self._config)

    def drain(self) -> None:
        """Process every remaining inbox entry synchronously, in arrival order."""
        if self._failed:
            return
        while True:
            with self._lock:
                if not self._inbox:
                    return
                port_name, sample = self._inbox.popleft()
            self._invoke(port_name, sample)

    def update_config(self, key: str, value) -> None:
        """Hot-update a config value (thread-safe for simple types under GIL)."""
        param_type = next(
            (p.type for p in self._spec.params if p.name == key), None
        )
        coerce = _PARAM_COERCE.get(param_type)
        if coerce and not isinstance(value, coerce):
            value = coerce(value)
        self._config[key] = value
        log.info("config: %s.%s = %r", self.node_id, key, value)

    def queue_depth(self) -> int:
        return len(self._inbox)

    def backlog_depth(self) -> int:
        return len(self._inbox)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class Pipeline:
    """Reactive DAG execution engine: sources push, pool executes downstream."""

    def __init__(self, max_workers: int = 4):
        self._nodes: dict[str, NodeInstance] = {}
        self._connections: list[Connection] = []
        self._clock = MasterClock()
        self._metrics = MetricsCollector()
        self._mode = PipelineMode.STOPPED
        self._max_workers = max_workers
        self._recorder: SessionRecorder | None = None
        self.on_sample: Callable[[str, str, Sample], None] | None = None
        # Reactive scheduling
        self._adjacency: dict[tuple[str, str], list[Connection]] = {}
        self._pool: ThreadPoolExecutor | None = None
        self._in_flight: set[str] = set()
        self._lock = threading.Lock()
        self._condition = threading.Condition(self._lock)

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
            log.warning("connect: node '%s' has no output port '%s'", src_id, src_port)
            raise ValueError(f"Node '{src_id}' has no output port '{src_port}'")

        dst_port_type = None
        for port in dst_node._spec.inputs:
            if port.name == dst_port:
                dst_port_type = port.type
                break
        if dst_port_type is None:
            log.warning("connect: node '%s' has no input port '%s'", dst_id, dst_port)
            raise ValueError(f"Node '{dst_id}' has no input port '{dst_port}'")

        if not compatible(src_port_type, dst_port_type):
            log.warning("connect: incompatible ports %s.%s (%s) → %s.%s (%s)",
                        src_id, src_port, src_port_type.__name__,
                        dst_id, dst_port, dst_port_type.__name__)
            raise IncompatiblePortError(
                f"Cannot connect {src_port_type.__name__} to {dst_port_type.__name__}"
            )

        self._connections.append(Connection(
            src_id=src_id, src_port=src_port, dst_id=dst_id, dst_port=dst_port,
        ))
        log.info("connected %s.%s -> %s.%s", src_id, src_port, dst_id, dst_port)

    def _build_adjacency(self) -> None:
        """Build precomputed adjacency list: (src_id, port_name) -> [Connection]."""
        self._adjacency.clear()
        for conn in self._connections:
            key = (conn.src_id, conn.src_port)
            self._adjacency.setdefault(key, []).append(conn)

    def _dispatch(self, src_id: str, outputs: dict[str, Sample]) -> None:
        """Route outputs from a node to all connected downstream nodes."""
        for port_name, sample in outputs.items():
            downstream = self._adjacency.get((src_id, port_name), ())
            recorder = self._recorder
            if recorder:
                recorder.write(sample, node_id=src_id)
            if self.on_sample:
                try:
                    self.on_sample(src_id, port_name, sample)
                except Exception:
                    log.exception("on_sample callback raised for %s.%s", src_id, port_name)
            log.debug("dispatch: %s.%s → %d downstream (fid=%d)", src_id, port_name, len(downstream), sample.frame_id)
            for conn in downstream:
                copy = sample.replace(metadata=dict(sample.metadata))
                self._nodes[conn.dst_id].on_input(conn.dst_port, copy)

    def _schedule_node(self, node: NodeInstance) -> None:
        """Pop the next inbox entry and submit to the pool, one at a time per node."""
        if node._failed:
            return
        if self._mode != PipelineMode.LIVE:
            return
        with self._lock:
            if node.node_id in self._in_flight:
                return
            if not node._inbox:
                return
            port_name, sample = node._inbox.popleft()
            self._in_flight.add(node.node_id)
        log.debug("schedule: %s.%s fid=%d", node.node_id, port_name, sample.frame_id)
        if self._pool:
            self._pool.submit(self._execute_node, node, port_name, sample)

    def _execute_node(self, node: NodeInstance, port_name: str, sample: Sample) -> None:
        """Run in pool thread: invoke func with single (port, sample), then re-schedule."""
        try:
            node._invoke(port_name, sample)
        finally:
            with self._lock:
                self._in_flight.discard(node.node_id)
            log.debug("executed: %s", node.node_id)
            # Re-check: new arrivals may be queued while we were executing
            self._schedule_node(node)

    def _topological_sort(self) -> list[str]:
        """Kahn's algorithm on the connection graph. Returns node_ids in topo order."""
        in_degree: dict[str, int] = {nid: 0 for nid in self._nodes}
        children: dict[str, set[str]] = {nid: set() for nid in self._nodes}
        for conn in self._connections:
            if conn.dst_id in in_degree:
                in_degree[conn.dst_id] += 1
            if conn.src_id in children:
                children[conn.src_id].add(conn.dst_id)

        queue = deque(nid for nid, deg in in_degree.items() if deg == 0)
        result = []
        while queue:
            nid = queue.popleft()
            result.append(nid)
            for child in children.get(nid, ()):
                in_degree[child] -= 1
                if in_degree[child] == 0:
                    queue.append(child)
        return result

    def _drain_backlogs(self) -> None:
        """Drain all inboxes in topological order."""
        order = self._topological_sort()
        for node_id in order:
            node = self._nodes[node_id]
            if node._spec.kind != "source":
                depth = node.queue_depth()
                if depth:
                    log.debug("draining %s: %d inbox", node_id, depth)
                node.drain()

    def start(self) -> None:
        self._clock.start()
        self._mode = PipelineMode.LIVE

        # Build adjacency list
        self._build_adjacency()

        # Track which ports are actually connected
        for node_id, node in self._nodes.items():
            node._connected_ports = {c.dst_port for c in self._connections if c.dst_id == node_id}

        # Init all nodes (failures are logged but don't kill the pipeline)
        for node in self._nodes.values():
            try:
                node.init()
            except Exception:
                log.error("init failed for '%s', marking as failed", node.node_id, exc_info=True)
                node._failed = True

        # Start thread pool for non-source nodes
        self._pool = ThreadPoolExecutor(
            max_workers=self._max_workers,
            thread_name_prefix="sigflow-pool",
        )
        self._in_flight.clear()

        # Start source threads last (they push data)
        for node in self._nodes.values():
            if node._spec.kind == "source" and not node._failed:
                node.start()

        log.info("pipeline started: %d nodes, %d connections, pool=%d",
                 len(self._nodes), len(self._connections), self._max_workers)

    def stop(self) -> None:
        log.info("stopping pipeline ...")
        # Stop sources first
        for node in self._nodes.values():
            if node._spec.kind == "source":
                node.stop()

        time.sleep(0.05)

        # Drain backlogs in topological order
        self._mode = PipelineMode.DRAINING
        self._drain_backlogs()

        # Shutdown pool
        if self._pool:
            self._pool.shutdown(wait=True)
            self._pool = None

        if self._recorder:
            self._recorder.finalize()
            self._recorder = None

        for node in self._nodes.values():
            if not node._failed:
                try:
                    node.cleanup()
                except Exception:
                    log.exception("cleanup failed for '%s', continuing", node.node_id)

        self._mode = PipelineMode.STOPPED
        log.info("pipeline stopped")

    def start_recording(self, output_dir="recordings",
                        on_sample=None) -> Path | None:
        """Start recording all dispatched samples. Returns session dir."""
        self._recorder = SessionRecorder(output_dir, on_sample=on_sample)
        return self._recorder.session_dir

    def stop_recording(self):
        """Stop recording and finalize the session. Returns session dir Path or None."""
        if self._recorder:
            session_dir = self._recorder.finalize()
            self._recorder = None
            log.info("recording stopped")
            return session_dir
        return None

    def drain(self) -> None:
        self._mode = PipelineMode.DRAINING
        self._drain_backlogs()

    def update_node_config(self, node_id: str, key: str, value) -> None:
        """Hot-update a config value on a running node."""
        node = self._nodes.get(node_id)
        if node:
            node.update_config(key, value)
        else:
            log.warning("update_node_config: node '%s' not in pipeline", node_id)

    @property
    def node_ids(self) -> list[str]:
        """Return all node IDs in the pipeline."""
        return list(self._nodes.keys())

    def get_node_state(self, node_id: str) -> dict | None:
        """Return the mutable state dict for a node, or None if not found.

        Useful for injecting callbacks or shared objects (e.g. text providers)
        into node state before or after pipeline start.
        """
        node = self._nodes.get(node_id)
        return node._state if node else None

    def metrics_snapshot(self) -> dict:
        snapshots = {}
        for node_id, node in self._nodes.items():
            if node._metrics:
                # Sync custom metrics from node state
                for key in ("_tok_s", "_drops"):
                    val = node._state.get(key)
                    if val is not None:
                        node._metrics.set_custom(key.lstrip("_"), val)
                snapshots[node_id] = node._metrics.snapshot(
                    queue_depth=node.queue_depth(),
                    backlog_depth=node.backlog_depth(),
                )
        return snapshots

    @classmethod
    def from_graph(cls, graph, **kwargs) -> Pipeline:
        p = cls(**kwargs)
        for node_def in graph.nodes:
            p.add_node(node_def.id, node_def.type, node_def.config)
        for conn in graph.connections:
            p.connect(conn.src_id, conn.src_port, conn.dst_id, conn.dst_port)
        return p
