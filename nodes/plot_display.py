"""Live plot display sink node.

Renders 1D signals as waveforms and 2D data as colormapped heatmaps,
then pushes the resulting image through the cv2 display queue.
No matplotlib GUI needed — just numpy + cv2 colormap.
"""
import cv2
import numpy as np

from sigflow.node import sink_node, Param
from sigflow.nodes.cv2_display import _display_queue
from sigflow.types import Port, TimeSeries


def _render_1d(data: np.ndarray, width: int = 640, height: int = 480) -> np.ndarray:
    """Render a 1D signal as a waveform image."""
    img = np.zeros((height, width, 3), dtype=np.uint8)

    if len(data) == 0:
        return img

    # Normalize to [0, 1]
    dmin, dmax = data.min(), data.max()
    if dmax - dmin > 0:
        normalized = (data - dmin) / (dmax - dmin)
    else:
        normalized = np.full_like(data, 0.5)

    # Map to pixel coordinates
    x_coords = np.linspace(0, width - 1, len(normalized)).astype(np.int32)
    y_coords = (height - 1 - (normalized * (height - 1))).astype(np.int32)

    # Draw connected line segments
    pts = np.column_stack([x_coords, y_coords]).reshape(-1, 1, 2)
    cv2.polylines(img, [pts], isClosed=False, color=(0, 255, 0), thickness=1)

    return img


def _render_2d(data: np.ndarray) -> np.ndarray:
    """Render a 2D array as a colormapped heatmap image."""
    if data.size == 0:
        return np.zeros((480, 640, 3), dtype=np.uint8)

    # Normalize to 0-255
    dmin, dmax = data.min(), data.max()
    if dmax - dmin > 0:
        normalized = ((data - dmin) / (dmax - dmin) * 255).astype(np.uint8)
    else:
        normalized = np.zeros_like(data, dtype=np.uint8)

    # Flip vertically (origin at bottom) and apply colormap
    flipped = np.flipud(normalized)
    colored = cv2.applyColorMap(flipped, cv2.COLORMAP_VIRIDIS)

    # Resize to a reasonable display size
    h, w = colored.shape[:2]
    scale = max(1, 480 // max(h, 1))
    if scale > 1:
        colored = cv2.resize(colored, (w * scale, h * scale), interpolation=cv2.INTER_NEAREST)

    return colored


@sink_node(
    name="plot_display",
    inputs=[Port("signal", TimeSeries)],
    category="display",
    params=[
        Param("window_name", "str", "sigflow plot", label="Window Name"),
    ],
)
def plot_display(item, *, state, config):
    window_name = config["window_name"]
    data = item.data

    if data.ndim == 1:
        img = _render_1d(data)
    else:
        img = _render_2d(data)

    _display_queue.append((window_name, img))
