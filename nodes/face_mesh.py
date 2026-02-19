"""MediaPipe face mesh detection node.

Runs FaceLandmarker on each frame and outputs all 468 normalized
landmarks as a (468, 3) numpy array [x, y, z].  Downstream nodes
(face_roi, mesh_overlay) consume these landmarks.
"""
import logging
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

from sigflow.node import process_node
from sigflow.types import Port, TimeSeries2D, FaceLandmarks

log = logging.getLogger(__name__)

_MODEL_PATH = str(Path(__file__).resolve().parents[3] / "weights" / "face_landmarker.task")


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

    result = state["landmarker"].detect(mp_image)

    if not result.face_landmarks:
        return None

    lms = result.face_landmarks[0]
    coords = np.array([[lm.x, lm.y, lm.z] for lm in lms], dtype=np.float32)

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
    base_options = mp.tasks.BaseOptions(model_asset_path=_MODEL_PATH)
    options = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        num_faces=1,
        min_face_detection_confidence=0.5,
    )
    state["landmarker"] = mp.tasks.vision.FaceLandmarker.create_from_options(options)
    log.info("initialized MediaPipe FaceLandmarker (model=%s)", _MODEL_PATH)


@face_mesh.cleanup
def face_mesh_cleanup(state, config):
    if "landmarker" in state:
        state["landmarker"].close()
        log.info("closed MediaPipe FaceLandmarker")
