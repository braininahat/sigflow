"""Generic 2D crop process node."""
from sigflow.node import process_node
from sigflow.types import Port, TimeSeries2D


@process_node(
    name="crop",
    inputs=[Port("frame", TimeSeries2D)],
    outputs=[Port("frame", TimeSeries2D)],
    category="transform",
)
def crop(item, *, state, config):
    x, y, w, h = config["x"], config["y"], config["w"], config["h"]
    cropped = item.data[y:y + h, x:x + w]
    return {"frame": item.replace(data=cropped)}
