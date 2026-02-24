"""Generic 2D crop process node."""
import logging

from sigflow.node import process_node, Param
from sigflow.types import Port, TimeSeries2D

log = logging.getLogger(__name__)


@process_node(
    name="crop",
    inputs=[Port("frame", TimeSeries2D)],
    outputs=[Port("frame", TimeSeries2D)],
    category="transform",
    params=[
        Param("x", "int", 0, label="X Offset", min=0, max=4096),
        Param("y", "int", 0, label="Y Offset", min=0, max=4096),
        Param("w", "int", 320, label="Width", min=1, max=4096),
        Param("h", "int", 240, label="Height", min=1, max=4096),
    ],
)
def crop(item, *, state, config):
    x, y, w, h = config["x"], config["y"], config["w"], config["h"]
    cropped = item.data[y:y + h, x:x + w]
    return {"frame": item.replace(data=cropped)}
