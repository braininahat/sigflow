"""Region bounding box from face landmarks.

Consumes FaceLandmarks (468×3 normalized array from face_mesh node),
selects a landmark group (mouth, eyes, nose, face oval), and outputs
the enclosing bounding box as an ROI with configurable padding.
"""
import numpy as np

from sigflow.node import process_node, Param
from sigflow.types import Port, FaceLandmarks, ROI

# Canonical MediaPipe FaceMesh landmark indices per region.
# Derived from mp.solutions.face_mesh FACEMESH_LIPS, FACEMESH_*_EYE, etc.
_LANDMARK_GROUPS = {
    "mouth": [
        0, 13, 14, 17, 37, 39, 40, 61, 78, 80, 81, 82, 84, 87, 88, 91,
        95, 146, 178, 181, 185, 191, 267, 269, 270, 291, 308, 310, 311,
        312, 314, 317, 318, 321, 324, 375, 402, 405, 409, 415,
    ],
    "left_eye": [
        249, 263, 362, 373, 374, 380, 381, 382, 384, 385, 386, 387,
        388, 390, 398, 466,
    ],
    "right_eye": [
        7, 33, 133, 144, 145, 153, 154, 155, 157, 158, 159, 160,
        161, 163, 173, 246,
    ],
    "nose": [
        1, 2, 3, 4, 5, 6, 48, 64, 98, 115, 168, 195, 197, 220,
        278, 294, 327, 344, 440,
    ],
    "face_oval": [
        10, 21, 54, 58, 67, 93, 103, 109, 127, 132, 136, 148, 149,
        150, 152, 162, 172, 176, 234, 251, 284, 288, 297, 323, 332,
        338, 356, 361, 365, 377, 378, 379, 389, 397, 400, 454,
    ],
}


def compute_roi(landmarks, indices, frame_h, frame_w, padding):
    """Compute padded bounding box from selected landmark indices.

    Args:
        landmarks: (N, 3) numpy array of normalized [x, y, z] coords
        indices: list of landmark indices to enclose
        frame_h, frame_w: frame dimensions for denormalization
        padding: fraction of bbox size to add on each side

    Returns:
        numpy array [x, y, w, h] in pixel coordinates, clamped to frame bounds.
    """
    selected = landmarks[indices]
    xs = selected[:, 0] * frame_w
    ys = selected[:, 1] * frame_h

    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()

    w = x_max - x_min
    h = y_max - y_min

    # Add padding
    pad_x = w * padding
    pad_y = h * padding
    x_min -= pad_x
    y_min -= pad_y
    w += 2 * pad_x
    h += 2 * pad_y

    # Clamp to frame bounds
    x = max(0, round(x_min))
    y = max(0, round(y_min))
    w = min(round(w), frame_w - x)
    h = min(round(h), frame_h - y)

    return np.array([x, y, w, h], dtype=np.int32)


@process_node(
    name="face_roi",
    inputs=[Port("landmarks", FaceLandmarks)],
    outputs=[Port("roi", ROI)],
    category="detection",
    params=[
        Param("region", "choice", "mouth", label="Region",
              choices=list(_LANDMARK_GROUPS.keys())),
        Param("padding", "float", 0.2, label="Padding", min=0.0, max=1.0),
    ],
)
def face_roi(item, *, state, config):
    landmarks = item.data  # (468, 3) normalized
    frame_shape = item.metadata.get("frame_shape")
    if frame_shape is None:
        return None
    h, w = frame_shape

    indices = _LANDMARK_GROUPS[config["region"]]
    roi = compute_roi(landmarks, indices, h, w, config["padding"])

    return {
        "roi": item.replace(
            data=roi,
            port_type=ROI,
            metadata={**item.metadata, "region": config["region"]},
        ),
    }
