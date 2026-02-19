"""Keypoints overlay process node.

Multi-input node: buffers a frame and keypoints in state,
draws keypoint circles on the frame when both are available.
"""
import cv2
import numpy as np

from sigflow.node import process_node, Param
from sigflow.types import Port, TimeSeries2D, Keypoints


@process_node(
    name="keypoints_overlay",
    inputs=[Port("frame", TimeSeries2D), Port("keypoints", Keypoints)],
    outputs=[Port("overlay", TimeSeries2D)],
    category="display",
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

    frame = state["frame"].data.copy()
    kps = state["keypoints"].data  # (num_joints, 3): [x, y, confidence]
    threshold = config["confidence_threshold"]
    radius = config["radius"]

    # Scale keypoints if frame dimensions differ from inference input
    frame_h, frame_w = frame.shape[:2]
    infer_shape = state["keypoints"].metadata.get("frame_shape")
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
            cv2.circle(frame, (px, py), radius, (0, 255, 0), -1)

    return {"overlay": state["frame"].replace(data=frame)}
