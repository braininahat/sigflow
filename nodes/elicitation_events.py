"""Elicitation event source node.

Polls a shared queue for events from ElicitationService and routes them
to the appropriate output port: LLM prompt requests, recorded audio for
scoring, and LSL event markers.

Replaces the old elicitation_markers.py with multi-port output support.
"""
from __future__ import annotations

import json
import logging
import queue as queue_mod
import time

from sigflow.node import source_node
from sigflow.types import AudioSignal, Event, MarkerEvent, Port, Sample, TimeSeries1D

log = logging.getLogger(__name__)

# Module-level queue — ElicitationService pushes, source node polls.
_event_queue: queue_mod.Queue | None = None


def set_event_queue(q: queue_mod.Queue | None) -> None:
    """Set the shared queue. Called by ElicitationService.start()."""
    global _event_queue
    _event_queue = q


@source_node(
    name="elicitation_events",
    outputs=[
        Port("prompt", Event),
        Port("audio", AudioSignal),
        Port("markers", MarkerEvent),
    ],
    category="elicitation",
)
def elicitation_events(*, state, config, clock):
    if _event_queue is None:
        time.sleep(0.1)
        return None

    try:
        event = _event_queue.get(timeout=0.05)
    except queue_mod.Empty:
        return None

    event_type = event.get("type", "marker")
    source_id = config.get("source_id", "elicitation")
    ts = clock.lsl_now()
    session_ms = clock.session_time_ms()

    if event_type == "prompt":
        # LLM prompt request: data = {"system": str, "user": str, "max_tokens": int}
        return {"prompt": Sample(
            source_id=source_id,
            lsl_timestamp=ts,
            session_time_ms=session_ms,
            data=event.get("data", {}),
            port_type=Event,
        )}
    elif event_type == "audio":
        # Recorded speech for phoneme scoring
        return {"audio": Sample(
            source_id=source_id,
            lsl_timestamp=ts,
            session_time_ms=session_ms,
            data=event.get("data"),
            metadata=event.get("metadata", {}),
            port_type=AudioSignal,
        )}
    else:
        # All other events → marker stream (for XDF recording)
        return {"markers": Sample(
            source_id=source_id,
            lsl_timestamp=ts,
            session_time_ms=session_ms,
            data=json.dumps(event),
            port_type=MarkerEvent,
        )}
