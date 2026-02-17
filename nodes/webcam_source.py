"""Webcam capture source node (cv2.VideoCapture)."""
import cv2

from sigflow.node import source_node
from sigflow.types import Port, Sample, CameraFrame


@source_node(
    name="webcam",
    outputs=[Port("frame", CameraFrame)],
    category="source",
)
def webcam(*, state, config, clock):
    if "cap" not in state:
        state["cap"] = cv2.VideoCapture(config.get("device", 0))
    ret, frame = state["cap"].read()
    if ret:
        return {"frame": Sample(
            source_id=config.get("source_id", "webcam"),
            lsl_timestamp=clock.lsl_now(),
            session_time_ms=clock.session_time_ms(),
            data=frame,
            metadata={},
            port_type=CameraFrame,
        )}
    return None


@webcam.cleanup
def webcam_cleanup(state, config):
    if "cap" in state:
        state["cap"].release()
