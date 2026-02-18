"""OpenCV window display sink node.

All cv2 GUI calls are serialized through a frame queue that must be pumped
from the main thread (via drain_display_queue), because GTK/HighGUI is not
thread-safe for concurrent window operations.
"""
import threading
from collections import deque

import cv2

from sigflow.node import sink_node, Param
from sigflow.types import Port, TimeSeries2D

# Shared queue: sink threads push (window_name, frame), main thread pops and displays
_display_queue: deque[tuple[str, object]] = deque()
_display_lock = threading.Lock()


def drain_display_queue() -> bool:
    """Call from the main thread to display queued frames. Returns True if any were shown."""
    shown = False
    with _display_lock:
        while _display_queue:
            window_name, frame = _display_queue.popleft()
            cv2.imshow(window_name, frame)
            shown = True
    if shown:
        cv2.waitKey(1)
    return shown


@sink_node(
    name="cv2_display",
    inputs=[Port("frame", TimeSeries2D)],
    category="display",
    params=[
        Param("window_name", "str", "sigflow", label="Window Name"),
    ],
)
def cv2_display(item, *, state, config):
    _display_queue.append((config["window_name"], item.data))
