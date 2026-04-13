"""Keypoints overlay process node.

Multi-input node: buffers a frame and keypoints in state,
draws keypoint circles on the frame when both are available.

Color coding by anatomical region (BGR for cv2):

Tongue (posterior → anterior warm gradient):
  vallecula, tongueRoot    → blue/teal  (deep structures)
  tongueBody               → green      (mid tongue)
  tongueDorsum             → yellow     (upper surface)
  tongueBlade              → orange     (front portion)
  tongueTip                → red/coral  (most anterior, clinically salient)
  hyoid, mandible, thyroid → lavender   (bony landmarks)
  shortTendon              → lavender

Lips:
  leftLip, rightLip        → coral      (commissures)
  top*inner                → rose       (upper inner)
  bottom*inner             → magenta    (lower inner)
  bottom*outer             → purple     (lower outer)
"""
import cv2
import numpy as np

from sigflow.node import process_node, Param
from sigflow.types import Port, TimeSeries2D, Keypoints

# BGR colors keyed by joint name prefix (longest prefix wins)
_JOINT_COLORS: dict[str, tuple[int, int, int]] = {
    # Tongue — posterior to anterior warm gradient
    "vallecula":    (200, 120, 60),   # teal-blue
    "tongueRoot":   (190, 160, 50),   # cyan-teal
    "tongueBody":   (80, 200, 80),    # green
    "tongueDorsum": (40, 210, 210),   # yellow
    "tongueBlade":  (40, 160, 240),   # orange
    "tongueTip":    (80, 100, 255),   # red-coral
    # Bony landmarks
    "hyoid":        (200, 160, 200),  # lavender
    "mandible":     (200, 160, 200),
    "shortTendon":  (200, 160, 200),
    "thyroid":      (200, 160, 200),
    # Lips
    "leftLip":      (100, 120, 240),  # coral
    "rightLip":     (100, 120, 240),
    "topleft":      (140, 100, 220),  # rose
    "topmid":       (140, 100, 220),
    "topright":     (140, 100, 220),
    "bottomlefti":  (180, 80, 200),   # magenta (inner)
    "bottommidi":   (180, 80, 200),
    "bottomrighti": (180, 80, 200),
    "bottomlefto":  (180, 120, 160),  # purple (outer)
    "bottommido":   (180, 120, 160),
    "bottomrighto": (180, 120, 160),
}

_DEFAULT_COLOR = (0, 255, 0)  # green fallback


def _build_color_lut(joint_names: list[str]) -> list[tuple[int, int, int]]:
    """Build a per-joint color lookup table from joint names via prefix matching."""
    colors = []
    for name in joint_names:
        matched = _DEFAULT_COLOR
        best_len = 0
        for prefix, color in _JOINT_COLORS.items():
            if name.startswith(prefix) and len(prefix) > best_len:
                matched = color
                best_len = len(prefix)
        colors.append(matched)
    return colors


@process_node(
    name="keypoints_overlay",
    inputs=[Port("frame", TimeSeries2D), Port("keypoints", Keypoints)],
    outputs=[Port("overlay", TimeSeries2D)],
    category="visualization",
    params=[
        Param("confidence_threshold", "float", 0.1,
              label="Min Confidence", min=0.0, max=1.0),
        Param("radius", "int", 4, label="Point Radius", min=1, max=20),
    ],
)
def keypoints_overlay(item, *, state, config):
    # Disambiguate which input port by checking port_type
    if issubclass(item.port_type, Keypoints):
        state["keypoints"] = item
    else:
        state["frame"] = item

    # Only produce output when both inputs are available
    if "frame" not in state or "keypoints" not in state:
        return None

    frame_sample = state.pop("frame")
    kps_sample = state.pop("keypoints")
    frame = frame_sample.data.copy()
    kps = kps_sample.data  # (num_joints, 3): [x, y, confidence]
    threshold = config["confidence_threshold"]
    radius = config["radius"]

    # Build color LUT once from joint names (cached in state)
    joint_names = kps_sample.metadata.get("joint_names", [])
    if joint_names and state.get("_color_lut_names") != joint_names:
        state["_color_lut"] = _build_color_lut(joint_names)
        state["_color_lut_names"] = joint_names
    color_lut = state.get("_color_lut")

    # Scale keypoints if frame dimensions differ from inference input
    frame_h, frame_w = frame.shape[:2]
    infer_shape = kps_sample.metadata.get("frame_shape")
    if infer_shape and (infer_shape[0] != frame_h or infer_shape[1] != frame_w):
        scale_x = frame_w / infer_shape[1]
        scale_y = frame_h / infer_shape[0]
    else:
        scale_x = scale_y = 1.0

    for i in range(kps.shape[0]):
        x, y, conf = kps[i]
        if conf >= threshold:
            px = int(x * scale_x)
            py = int(y * scale_y)
            color = color_lut[i] if color_lut and i < len(color_lut) else _DEFAULT_COLOR
            cv2.circle(frame, (px, py), radius, color, -1)

    return {"overlay": frame_sample.replace(data=frame)}
