"""DeepLabCut ONNX inference process nodes.

Two named palette entries — dlc_lips and dlc_tongue — with shared
preprocessing/postprocessing and a common inference body.

Preprocessing matches the C++ InferenceEngine implementation:
1. BGR uint8 → float32
2. Resize to nearest multiple of stride (8)
3. Subtract ImageNet mean [123.68, 116.779, 103.939]
4. NHWC layout [1, H, W, 3]

Postprocessing: heatmap peak detection via cv2.minMaxLoc,
coordinate scaling by stride.
"""
import logging

import time

import numpy as np
import yaml

from sigflow.node import process_node, Param
from sigflow.types import Port, TimeSeries2D, Keypoints

log = logging.getLogger(__name__)

# ImageNet mean pixel values (BGR order, matching OpenCV)
_IMAGENET_MEAN = np.array([103.939, 116.779, 123.68], dtype=np.float32)


def preprocess_frame(frame: np.ndarray, stride: int) -> np.ndarray:
    """Preprocess a BGR uint8 frame for DLC inference.

    Returns NHWC float32 tensor [1, H, W, 3] with dimensions
    rounded up to the nearest multiple of stride.
    """
    import cv2
    h, w = frame.shape[:2]
    target_h = ((h + stride - 1) // stride) * stride
    target_w = ((w + stride - 1) // stride) * stride

    if h != target_h or w != target_w:
        frame = cv2.resize(frame, (target_w, target_h))

    tensor = frame.astype(np.float32) - _IMAGENET_MEAN
    return tensor[np.newaxis]  # [1, H, W, 3]


def postprocess_heatmaps(
    heatmaps: np.ndarray,
    stride: float,
    num_joints: int,
    locref: np.ndarray | None = None,
) -> np.ndarray:
    """Extract keypoint coordinates from DLC heatmap output.

    Args:
        heatmaps: NHWC array [1, H/stride, W/stride, num_joints]
        stride: DLC output stride (typically 8.0)
        num_joints: Number of keypoints
        locref: Optional location refinement array [1, H/s, W/s, num_joints*2].
                Channels j*2 and j*2+1 are X and Y offsets scaled by 0.25.

    Returns:
        Array of shape (num_joints, 3) with columns [x, y, confidence].
        Coordinates are in pixel space of the preprocessed input.
    """
    import cv2
    hm = heatmaps[0]  # Remove batch dim → [H, W, J]
    lr = locref[0] if locref is not None else None
    keypoints = np.zeros((num_joints, 3), dtype=np.float32)

    for j in range(num_joints):
        channel = hm[:, :, j].copy()  # Contiguous for minMaxLoc
        _, max_val, _, max_loc = cv2.minMaxLoc(channel)
        # max_loc is (x, y) in heatmap space
        x = max_loc[0] * stride
        y = max_loc[1] * stride

        # Sub-pixel refinement from location reference output
        if lr is not None and lr.shape[2] >= (j + 1) * 2:
            x += lr[max_loc[1], max_loc[0], j * 2] * 0.25 * stride
            y += lr[max_loc[1], max_loc[0], j * 2 + 1] * 0.25 * stride

        keypoints[j, 0] = x
        keypoints[j, 1] = y
        keypoints[j, 2] = max_val
    return keypoints


def _load_model(state, config):
    """Lazy-load ONNX model and preprocessing config into state."""
    import onnxruntime as ort
    from sigflow.paths import resolve_data_path

    config_path = str(resolve_data_path(config["config_path"]))
    with open(config_path) as f:
        preprocess_cfg = yaml.safe_load(f)

    state["stride"] = int(preprocess_cfg.get("stride", 8))
    state["num_joints"] = int(preprocess_cfg["num_joints"])
    state["joint_names"] = preprocess_cfg.get("joint_names", [])
    state["model_input_width"] = int(preprocess_cfg.get("model_input_width", 0))

    from sigflow.onnx_providers import get_providers

    model_path = str(resolve_data_path(config["model_path"]))
    providers = get_providers()
    session = ort.InferenceSession(model_path, providers=providers)
    state["session"] = session
    state["input_name"] = session.get_inputs()[0].name

    active = session.get_providers()
    log.info(
        "loaded DLC model: %s (%d joints, stride=%d, providers=%s)",
        model_path, state["num_joints"], state["stride"], active,
    )


def _dlc_inference(item, *, state, config):
    if "session" not in state:
        try:
            _load_model(state, config)
        except Exception:
            log.error("failed to load DLC model: %s", config.get("model_path", "?"))
            raise

    frame = item.data
    stride = state["stride"]
    num_joints = state["num_joints"]
    model_input_width = state.get("model_input_width", 0)

    orig_h, orig_w = frame.shape[:2]

    # Resize to model input width if configured (e.g. tongue model needs 320px wide)
    if model_input_width > 0 and orig_w != model_input_width:
        import cv2
        aspect = orig_w / orig_h
        target_w = model_input_width
        target_h = ((int(target_w / aspect) + stride - 1) // stride) * stride
        frame = cv2.resize(frame, (target_w, target_h))

    t0 = time.perf_counter()
    tensor = preprocess_frame(frame, stride)
    outputs = state["session"].run(None, {state["input_name"]: tensor})
    heatmaps = outputs[0]  # [1, H/stride, W/stride, num_joints]
    locref = outputs[1] if len(outputs) > 1 else None  # [1, H/s, W/s, J*2]

    keypoints = postprocess_heatmaps(heatmaps, stride, num_joints, locref)

    # Remap keypoints back to original frame coordinates
    if model_input_width > 0 and orig_w != model_input_width:
        scale_x = orig_w / frame.shape[1]
        scale_y = orig_h / frame.shape[0]
        keypoints[:, 0] *= scale_x
        keypoints[:, 1] *= scale_y

    elapsed_ms = (time.perf_counter() - t0) * 1000
    state.setdefault("_inference_times", []).append(elapsed_ms)
    if len(state["_inference_times"]) % 100 == 0:
        avg = sum(state["_inference_times"][-100:]) / 100
        log.debug("dlc inference: avg %.1fms over last 100 frames", avg)

    return {
        "keypoints": item.replace(
            data=keypoints,
            port_type=Keypoints,
            metadata={
                **item.metadata,
                "joint_names": state["joint_names"],
                "frame_shape": (orig_h, orig_w),
            },
        ),
        "frame": item,
    }


@process_node(
    name="dlc_lips",
    inputs=[Port("frame", TimeSeries2D)],
    outputs=[Port("frame", TimeSeries2D), Port("keypoints", Keypoints)],
    category="inference",
    params=[
        Param("model_path", "str", "weights/lips.onnx", label="Model Path"),
        Param("config_path", "str", "weights/lips.preprocessing.yaml", label="Config Path"),
    ],
)
def dlc_lips(item, *, state, config):
    return _dlc_inference(item, state=state, config=config)


@process_node(
    name="dlc_tongue",
    inputs=[Port("frame", TimeSeries2D)],
    outputs=[Port("frame", TimeSeries2D), Port("keypoints", Keypoints)],
    category="inference",
    params=[
        Param("model_path", "str", "weights/tongue.onnx", label="Model Path"),
        Param("config_path", "str", "weights/tongue.preprocessing.yaml", label="Config Path"),
    ],
)
def dlc_tongue(item, *, state, config):
    return _dlc_inference(item, state=state, config=config)
