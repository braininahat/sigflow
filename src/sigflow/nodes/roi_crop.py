"""Dynamic ROI crop process node.

Multi-input node: buffers a frame and ROI in state,
crops the frame to the ROI bounds when both are available.
"""
import logging

import numpy as np

from sigflow.node import process_node, Param
from sigflow.types import Port, TimeSeries2D, ROI

log = logging.getLogger(__name__)


@process_node(
    name="roi_crop",
    inputs=[Port("frame", TimeSeries2D), Port("roi", ROI)],
    outputs=[Port("frame", TimeSeries2D)],
    category="transform",
    params=[
        Param("fallback_x", "int", 0, label="Fallback X", min=0, max=4096),
        Param("fallback_y", "int", 0, label="Fallback Y", min=0, max=4096),
        Param("fallback_w", "int", 320, label="Fallback W", min=1, max=4096),
        Param("fallback_h", "int", 240, label="Fallback H", min=1, max=4096),
    ],
)
def roi_crop(item, *, state, config):
    if issubclass(item.port_type, ROI):
        state["roi"] = item
    else:
        state["frame"] = item

    if "frame" not in state:
        return None

    frame = state["frame"].data
    fh, fw = frame.shape[:2]

    if "roi" in state:
        x, y, w, h = state["roi"].data
    else:
        log.debug("roi_crop: using fallback ROI (no ROI received yet)")
        x = config["fallback_x"]
        y = config["fallback_y"]
        w = config["fallback_w"]
        h = config["fallback_h"]

    # Clamp to frame bounds
    x = max(0, min(int(x), fw - 1))
    y = max(0, min(int(y), fh - 1))
    w = max(1, min(int(w), fw - x))
    h = max(1, min(int(h), fh - y))

    cropped = frame[y:y + h, x:x + w]
    return {"frame": state["frame"].replace(data=cropped)}
