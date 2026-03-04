"""Elicitation event marker source node.

Polls a shared queue for events from ElicitationService,
emitting them as MarkerEvent samples into the pipeline DAG.
Events flow through Pipeline._dispatch() → SessionRecorder._record_event()
→ XDF string stream, giving LSL-timestamped event markers alongside all
other recorded streams.
"""
import json
import logging
import queue as queue_mod
import time

from sigflow.node import source_node
from sigflow.types import MarkerEvent, Port, Sample

log = logging.getLogger(__name__)

# Module-level queue — ElicitationService pushes, source node polls.
_marker_queue: queue_mod.Queue | None = None


def set_marker_queue(q: queue_mod.Queue | None) -> None:
    """Set the shared queue. Called by ElicitationService.start()."""
    global _marker_queue
    _marker_queue = q


@source_node(
    name="elicitation_markers",
    outputs=[Port("markers", MarkerEvent)],
    category="elicitation",
)
def elicitation_markers(*, state, config, clock):
    if _marker_queue is None:
        time.sleep(0.1)
        return None
    try:
        event = _marker_queue.get(timeout=0.05)
    except queue_mod.Empty:
        return None
    return {"markers": Sample(
        source_id=config.get("source_id", "elicitation"),
        lsl_timestamp=clock.lsl_now(),
        session_time_ms=clock.session_time_ms(),
        data=json.dumps(event),
        port_type=MarkerEvent,
    )}
