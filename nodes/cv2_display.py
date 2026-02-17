"""OpenCV window display sink node."""
import cv2

from sigflow.node import sink_node
from sigflow.types import Port, TimeSeries2D


@sink_node(
    name="cv2_display",
    inputs=[Port("frame", TimeSeries2D)],
    category="display",
)
def cv2_display(item, *, state, config):
    window_name = config.get("window_name", "sigflow")
    cv2.imshow(window_name, item.data)
    cv2.waitKey(1)
