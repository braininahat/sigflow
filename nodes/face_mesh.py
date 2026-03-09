"""MediaPipe face mesh detection node.

Runs FaceLandmarker on each frame and outputs all 468 normalized
landmarks as a (468, 3) numpy array [x, y, z].  Downstream nodes
(face_roi, mesh_overlay) consume these landmarks.
"""
import logging
import threading
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

from sigflow.node import process_node
from sigflow.paths import resolve_data_path
from sigflow.types import Port, TimeSeries2D, FaceLandmarks

log = logging.getLogger(__name__)

# Lock around MediaPipe detect + result extraction to prevent protobuf
# thread-safety crashes (PyTuple_GET_SIZE assertion) when ONNX/TFLite
# threads run concurrently and trigger GC during protobuf iteration.
_detect_lock = threading.Lock()


@process_node(
    name="face_mesh",
    inputs=[Port("frame", TimeSeries2D)],
    outputs=[Port("frame", TimeSeries2D), Port("landmarks", FaceLandmarks)],
    category="detection",
)
def face_mesh(item, *, state, config):
    frame = item.data
    h, w = frame.shape[:2]

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    with _detect_lock:
        result = state["landmarker"].detect(mp_image)
        if not result.face_landmarks:
            log.debug("no face detected")
            return None
        # Extract to plain Python ASAP, then release protobuf objects
        lms = result.face_landmarks[0]
        raw = [(lm.x, lm.y, lm.z) for lm in lms]
        del lms, result

    coords = np.array(raw, dtype=np.float32)

    return {
        "landmarks": item.replace(
            data=coords,
            port_type=FaceLandmarks,
            metadata={**item.metadata, "frame_shape": (h, w)},
        ),
        "frame": item,
    }


@face_mesh.init
def face_mesh_init(state, config):
    model_path = str(resolve_data_path("weights/face_landmarker.task"))
    base_options = mp.tasks.BaseOptions(model_asset_path=model_path)
    options = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        num_faces=1,
        min_face_detection_confidence=0.5,
    )
    state["landmarker"] = mp.tasks.vision.FaceLandmarker.create_from_options(options)
    log.info("initialized MediaPipe FaceLandmarker (model=%s)", model_path)


@face_mesh.cleanup
def face_mesh_cleanup(state, config):
    if "landmarker" in state:
        state["landmarker"].close()
        log.info("closed MediaPipe FaceLandmarker")
