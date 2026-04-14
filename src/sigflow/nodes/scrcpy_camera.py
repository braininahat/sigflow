"""Android camera capture source via scrcpy V4L2 sink.

Launches scrcpy in camera mode to stream from the Android device's
front or back camera to a V4L2 loopback device, then reads frames
with cv2.VideoCapture.

The V4L2 loopback device is created on init and destroyed on cleanup.
"""
import logging

from sigflow.node import source_node, Param
from sigflow.types import Port, Sample, CameraFrame
from sigflow.nodes._scrcpy import (
    create_v4l2_device, destroy_v4l2_device,
    launch_scrcpy, wait_for_capture, kill_scrcpy,
)

log = logging.getLogger(__name__)


@source_node(
    name="scrcpy_camera",
    outputs=[Port("frame", CameraFrame)],
    params=[
        Param("serial", "str", "", label="Device Serial"),
        Param("camera_facing", "choice", "front", label="Camera",
              choices=["front", "back"]),
        Param("camera_size", "str", "640x480", label="Resolution"),
        Param("codec", "choice", "h264", label="Codec", choices=["h264", "h265"]),
        Param("max_fps", "int", 90, label="Max FPS", min=0, max=120),
        Param("bitrate", "str", "4M", label="Bitrate"),
        Param("source_id", "str", "camera", label="Source ID"),
        Param("scrcpy_path", "str", "scrcpy", label="scrcpy Path"),
    ],
)
def scrcpy_camera(*, state, config, clock):
    if "cap" not in state:
        return None

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
        log.warning("scrcpy_camera frame drop")
    elif drops % 100 == 0:
        log.warning("scrcpy_camera frame drops: %d consecutive", drops)
    return None


@scrcpy_camera.init
def scrcpy_camera_init(state, config):
    log.info("=== scrcpy_camera init ===")
    state["v4l2_device"] = create_v4l2_device("sigflow_camera")

    args = [
        "--video-source", "camera",
        "--camera-facing", config["camera_facing"],
        "--camera-size", config["camera_size"],
        "--video-codec", config["codec"],
        "--video-bit-rate", config["bitrate"],
        "--no-audio",
        "--v4l2-sink", state["v4l2_device"],
    ]
    if config["max_fps"] > 0:
        args += ["--max-fps", str(config["max_fps"])]
    if config["serial"]:
        args += ["--serial", config["serial"]]

    state["proc"] = launch_scrcpy(args, config["scrcpy_path"])

    cap = wait_for_capture(state["v4l2_device"], state["proc"])
    if cap is None:
        kill_scrcpy(state["proc"])
        return

    state["cap"] = cap
    log.info("=== scrcpy_camera ready (%s) ===", config["camera_facing"])


@scrcpy_camera.cleanup
def scrcpy_camera_cleanup(state, config):
    if "cap" in state:
        state["cap"].release()
    if "proc" in state:
        kill_scrcpy(state["proc"])
    if "v4l2_device" in state:
        destroy_v4l2_device(state["v4l2_device"])
