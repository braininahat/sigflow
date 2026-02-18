"""Canvas display sink — stores latest rendered frame for in-editor display.

Unlike cv2_display which opens separate OpenCV windows, this node stores
frames in a module-level dict that the editor reads on the main thread
to update an embedded QLabel widget inside the node on the graph canvas.
"""
import cv2
import numpy as np

from sigflow.node import sink_node, Param
from sigflow.nodes.plot_display import _render_1d, _render_2d
from sigflow.types import Port, TimeSeries

# Latest frame per label, read by editor on main thread.
# Dict assignment is atomic under CPython GIL — no lock needed.
_canvas_frames: dict[str, np.ndarray] = {}


@sink_node(
    name="canvas_display",
    inputs=[Port("signal", TimeSeries)],
    category="display",
    params=[Param("label", "str", "output", label="Label")],
)
def canvas_display(item, *, state, config):
    data = item.data
    if data.ndim == 1:
        frame = _render_1d(data, width=320, height=240)
    elif data.ndim == 2 and data.dtype != np.uint8:
        frame = _render_2d(data)
    elif data.ndim == 2:
        frame = cv2.cvtColor(data, cv2.COLOR_GRAY2BGR)
    else:
        frame = data
    _canvas_frames[config["label"]] = frame
