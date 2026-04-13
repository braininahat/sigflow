"""Webcam capture source node (cv2.VideoCapture)."""
import logging
import sys

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
    import cv2
    _API = cv2.CAP_V4L2 if sys.platform == "linux" else cv2.CAP_ANY
    if "cap" not in state:
        dev = config["device"]
        log.info("opening camera device %d (backend=%s)", dev, _API)
        cap = cv2.VideoCapture(dev, _API)
        if not cap.isOpened():
            log.error("webcam: failed to open device %d", dev)
            state["cap"] = cap
            return None
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        backend = cap.getBackendName()
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        log.info("webcam opened: %dx%d @ %.1f fps (backend=%s)", w, h, fps, backend)
        state["cap"] = cap
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
