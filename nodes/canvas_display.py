"""Canvas display sink — stores latest rendered frame for in-editor display.

Unlike cv2_display which opens separate OpenCV windows, this node stores
frames in a module-level dict that the editor reads on the main thread
to update an embedded QLabel widget inside the node on the graph canvas.
"""
import cv2
import numpy as np

from sigflow.node import sink_node, Param
from sigflow.types import Port, TimeSeries

# Latest frame per label, read by editor on main thread.
# Dict assignment is atomic under CPython GIL — no lock needed.
_canvas_frames: dict[str, np.ndarray] = {}


def _render_1d(data: np.ndarray, width: int = 320, height: int = 240) -> np.ndarray:
    """Render a 1D signal as a waveform image."""
    img = np.zeros((height, width, 3), dtype=np.uint8)
    if len(data) == 0:
        return img
    dmin, dmax = data.min(), data.max()
    if dmax - dmin > 0:
        normalized = (data - dmin) / (dmax - dmin)
    else:
        normalized = np.full_like(data, 0.5)
    x_coords = np.linspace(0, width - 1, len(normalized)).astype(np.int32)
    y_coords = (height - 1 - (normalized * (height - 1))).astype(np.int32)
    pts = np.column_stack([x_coords, y_coords]).reshape(-1, 1, 2)
    cv2.polylines(img, [pts], isClosed=False, color=(0, 255, 0), thickness=1)
    return img


def _render_2d(data: np.ndarray) -> np.ndarray:
    """Render a 2D array as a colormapped heatmap image."""
    if data.size == 0:
        return np.zeros((240, 320, 3), dtype=np.uint8)
    dmin, dmax = data.min(), data.max()
    if dmax - dmin > 0:
        normalized = ((data - dmin) / (dmax - dmin) * 255).astype(np.uint8)
    else:
        normalized = np.zeros_like(data, dtype=np.uint8)
    flipped = np.flipud(normalized)
    colored = cv2.applyColorMap(flipped, cv2.COLORMAP_VIRIDIS)
    h, w = colored.shape[:2]
    scale = max(1, 240 // max(h, 1))
    if scale > 1:
        colored = cv2.resize(colored, (w * scale, h * scale), interpolation=cv2.INTER_NEAREST)
    return colored


@sink_node(
    name="canvas_display",
    inputs=[Port("signal", TimeSeries)],
    category="display",
    params=[Param("label", "str", "output", label="Label")],
)
def canvas_display(item, *, state, config):
    data = item.data
    if data.ndim == 1:
        frame = _render_1d(data)
    elif data.ndim == 2 and data.dtype != np.uint8:
        frame = _render_2d(data)
    elif data.ndim == 2:
        frame = cv2.cvtColor(data, cv2.COLOR_GRAY2BGR)
    else:
        frame = data
    _canvas_frames[config["_node_id"]] = frame
