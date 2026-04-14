"""Face mesh overlay visualization node.

Multi-input node: buffers a frame and face landmarks in state,
draws mesh connections on the frame when both are available.
"""
import numpy as np

from sigflow.node import process_node, Param
from sigflow.types import Port, TimeSeries2D, FaceLandmarks

# Lazily loaded from mediapipe on first use
_CONNECTION_SETS = {}


def _load_connections():
    """Load connection sets from mediapipe (called once)."""
    if _CONNECTION_SETS:
        return
    import mediapipe as mp
    C = mp.tasks.vision.FaceLandmarksConnections
    _CONNECTION_SETS["contours"] = [(c.start, c.end) for c in C.FACE_LANDMARKS_CONTOURS]
    _CONNECTION_SETS["tesselation"] = [(c.start, c.end) for c in C.FACE_LANDMARKS_TESSELATION]
    _CONNECTION_SETS["lips"] = [(c.start, c.end) for c in C.FACE_LANDMARKS_LIPS]
    _CONNECTION_SETS["eyes"] = (
        [(c.start, c.end) for c in C.FACE_LANDMARKS_LEFT_EYE]
        + [(c.start, c.end) for c in C.FACE_LANDMARKS_RIGHT_EYE]
        + [(c.start, c.end) for c in C.FACE_LANDMARKS_LEFT_IRIS]
        + [(c.start, c.end) for c in C.FACE_LANDMARKS_RIGHT_IRIS]
    )
    _CONNECTION_SETS["all"] = (
        _CONNECTION_SETS["tesselation"] + _CONNECTION_SETS["contours"]
    )


@process_node(
    name="mesh_overlay",
    inputs=[Port("frame", TimeSeries2D), Port("landmarks", FaceLandmarks)],
    outputs=[Port("overlay", TimeSeries2D)],
    category="visualization",
    params=[
        Param("style", "choice", "contours", label="Style",
              choices=["contours", "tesselation", "lips", "eyes", "all"]),
        Param("thickness", "int", 1, label="Line Thickness", min=1, max=5),
    ],
)
def mesh_overlay(item, *, state, config):
    if issubclass(item.port_type, FaceLandmarks):
        state["landmarks"] = item
        state["_no_face"] = item.metadata.get("no_face", False)
    else:
        state["frame"] = item

    if "frame" not in state or "landmarks" not in state:
        return None

    _load_connections()

    frame_sample = state.pop("frame")
    if state.pop("_no_face", False):
        state.pop("landmarks", None)
        return {"overlay": frame_sample}
    landmarks_data = state.pop("landmarks").data  # (468, 3) normalized
    frame = frame_sample.data.copy()
    h, w = frame.shape[:2]
    thickness = config["thickness"]

    connections = _CONNECTION_SETS.get(config["style"], _CONNECTION_SETS["contours"])

    # Convert normalized coords to pixel coords once
    px = (landmarks_data[:, 0] * w).astype(np.int32)
    py = (landmarks_data[:, 1] * h).astype(np.int32)

    import cv2
    for start, end in connections:
        cv2.line(frame, (px[start], py[start]), (px[end], py[end]),
                 (0, 255, 0), thickness)

    return {"overlay": frame_sample.replace(data=frame)}
