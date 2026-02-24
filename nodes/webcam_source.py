"""Webcam capture source node (cv2.VideoCapture)."""
import logging

import cv2

from sigflow.node import source_node, Param
from sigflow.types import Port, Sample, CameraFrame

log = logging.getLogger(__name__)


@source_node(
    name="webcam",
    outputs=[Port("frame", CameraFrame)],
    params=[
        Param("device", "int", 0, label="Device Index", min=0, max=10),
        Param("source_id", "str", "webcam", label="Source ID"),
    ],
)
def webcam(*, state, config, clock):
    if "cap" not in state:
        log.info("opening camera device %d", config["device"])
        state["cap"] = cv2.VideoCapture(config["device"])
        state["cap"].set(cv2.CAP_PROP_BUFFERSIZE, 1)
    ret, frame = state["cap"].read()
    if ret:
        state["_drop_count"] = 0
        return {"frame": Sample(
            source_id=config["source_id"],
            lsl_timestamp=clock.lsl_now(),
            session_time_ms=clock.session_time_ms(),
            data=frame,
            metadata={},
            port_type=CameraFrame,
        )}
    drops = state.get("_drop_count", 0) + 1
    state["_drop_count"] = drops
    if drops == 1:
        log.warning("webcam frame drop (device=%d)", config["device"])
    elif drops % 100 == 0:
        log.warning("webcam frame drops: %d consecutive (device=%d)", drops, config["device"])
    return None


@webcam.cleanup
def webcam_cleanup(state, config):
    if "cap" in state:
        log.info("releasing camera device")
        state["cap"].release()
