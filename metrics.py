"""Per-node metrics collection for sigflow pipelines."""
from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass


@dataclass
class NodeMetrics:
    """Snapshot of a node's performance metrics."""
    node_id: str = ""
    fps: float = 0.0
    avg_process_ms: float = 0.0
    peak_process_ms: float = 0.0
    queue_depth: int = 0
    items_processed: int = 0
    backlog_depth: int = 0
    last_update: float = 0.0


class MetricsTracker:
    """Per-node metrics tracker. Thread-safe."""

    def __init__(self, node_id: str, window_sec: float = 1.0):
        self._node_id = node_id
        self._window_sec = window_sec
        self._lock = threading.Lock()
        self._times: deque[tuple[float, float]] = deque()  # (timestamp, elapsed_ms)
        self._items_processed = 0

    def record(self, elapsed_ms: float) -> None:
        now = time.monotonic()
        with self._lock:
            self._times.append((now, elapsed_ms))
            self._items_processed += 1

    def _prune(self, now: float) -> None:
        cutoff = now - self._window_sec
        while self._times and self._times[0][0] < cutoff:
            self._times.popleft()

    def snapshot(self, queue_depth: int = 0, backlog_depth: int = 0) -> NodeMetrics:
        now = time.monotonic()
        with self._lock:
            self._prune(now)
            count = len(self._times)
            if count == 0:
                return NodeMetrics(
                    node_id=self._node_id,
                    queue_depth=queue_depth,
                    backlog_depth=backlog_depth,
                    items_processed=self._items_processed,
                    last_update=now,
                )
            elapsed_values = [t[1] for t in self._times]
            return NodeMetrics(
                node_id=self._node_id,
                fps=count / self._window_sec,
                avg_process_ms=sum(elapsed_values) / count,
                peak_process_ms=max(elapsed_values),
                queue_depth=queue_depth,
                backlog_depth=backlog_depth,
                items_processed=self._items_processed,
                last_update=now,
            )


class MetricsCollector:
    """Aggregates MetricsTrackers from all nodes in a pipeline."""

    def __init__(self):
        self._trackers: dict[str, MetricsTracker] = {}

    def create_tracker(self, node_id: str) -> MetricsTracker:
        tracker = MetricsTracker(node_id)
        self._trackers[node_id] = tracker
        return tracker

    def snapshot(self) -> dict[str, NodeMetrics]:
        return {nid: t.snapshot() for nid, t in self._trackers.items()}
