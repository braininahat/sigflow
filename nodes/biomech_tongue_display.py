"""Biomechanical tongue display sink — drives the MyoSim3D equilibrium solver
from DLC ultrasound tongue keypoints via the trained ridge-regression inverse
mapping.

Pipeline per frame (tracking phase):
    DLC 16 keypoints → slice dorsal[0:11]        (matches MyoSim3D
                                                  `vallecula → tongueTip2`)
                    → pixels × mm_per_pixel      (metadata)
                    → cubic spline fit + arc-length resample to 11 equidistant
                      (Option A from docs/plans/biomech-inverse-mapping.md —
                      the ridge was trained on arc-length resamples; DLC gives
                      anatomically-named points, which are not equally spaced)
                    → axis swap US(x, y) → model(Y, Z):
                        model_Y = -US_y ,  model_Z =  US_x
                    → displacement from per-patient calibration rest pose in
                      aligned (Procrustes rigid+scale) model space
                    → predict_activations(disp, W, b)   # 22-dim → 23-dim, 2 µs
                    → tongue_model.set_activations(acts)
                    → solve() (vectorized NumPy, ~60 ms on CPU)
                    → interleaved pos+normal vertex buffer
                    → _display_callback(display_id, "mesh", bytes)

Toggle:
    A single bool parameter `enabled` controls whether this sink emits mesh
    updates. Paired with the `enabled` flag on `tongue_model_display`, the
    Settings service flips both to switch which tongue rendering path feeds
    the Qt Quick 3D TongueMeshGeometry. Both nodes target the same display_id
    by default so the toggle is instantaneous.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from sigflow.node import Param, sink_node
from sigflow.types import Keypoints, Port

log = logging.getLogger(__name__)


# ─── Helpers ────────────────────────────────────────────────────────


def _spline_arc_resample(points_2d: np.ndarray, n_out: int = 11) -> np.ndarray:
    """Cubic spline fit through n anatomical points, resample to n_out
    arc-length-equidistant points.

    Matches the preprocessing the ridge regression was trained on.
    """
    from scipy.interpolate import CubicSpline

    n = len(points_2d)
    if n < 2:
        return points_2d
    # Cumulative chord length parameter
    diffs = np.diff(points_2d, axis=0)
    seg = np.linalg.norm(diffs, axis=1)
    t = np.concatenate([[0.0], np.cumsum(seg)])
    if t[-1] <= 1e-9:
        return np.repeat(points_2d[:1], n_out, axis=0)
    t /= t[-1]
    cs = CubicSpline(t, points_2d, axis=0)
    u = np.linspace(0.0, 1.0, n_out)
    return cs(u)


def _procrustes(src: np.ndarray, dst: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Rigid + uniform-scale alignment of src → dst (both (n, 2)).

    Returns (rotation, translation, scale) so that
        dst ≈ scale * (src @ rotation) + translation
    """
    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    src_c = src - src_mean
    dst_c = dst - dst_mean
    src_norm = float(np.sqrt((src_c ** 2).sum()))
    dst_norm = float(np.sqrt((dst_c ** 2).sum()))
    if src_norm < 1e-9 or dst_norm < 1e-9:
        return np.eye(2, dtype=np.float64), np.zeros(2, dtype=np.float64), 1.0
    U, _, Vt = np.linalg.svd(src_c.T @ dst_c)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        Vt[-1] *= -1
        R = U @ Vt
    scale = float((dst_c @ R.T * src_c).sum() / (src_norm ** 2)) if False else dst_norm / src_norm
    t = dst_mean - scale * (src_mean @ R)
    return R, t, scale


def _us_to_model_yz(points_mm: np.ndarray) -> np.ndarray:
    """Map ultrasound (x_mm, y_mm) → model (Y, Z) via the design doc axes:
    model_Y = -US_y, model_Z = US_x. Returns (n, 2).
    """
    out = np.empty_like(points_mm)
    out[:, 0] = -points_mm[:, 1]  # Y
    out[:, 1] = points_mm[:, 0]   # Z
    return out


def _build_tongue_model(model_path: Path):
    from sigflow.biomech.model import TongueModel

    return TongueModel(str(model_path))


def _load_or_build_mapping(cache_path: Path, s3d_path: Path):
    from sigflow.biomech.inverse import build_inverse_mapping, load_inverse_mapping
    from sigflow.biomech.s3d_parser import parse_s3d

    if cache_path.exists():
        log.info("biomech inverse: loading cached mapping from %s", cache_path)
        return load_inverse_mapping(str(cache_path))
    log.info("biomech inverse: building new mapping (5000 samples, ~3s on GPU)")
    model_raw = parse_s3d(str(s3d_path))
    mapping = build_inverse_mapping(model_raw, n_samples=5000, alpha=1.0, seed=0)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    from sigflow.biomech.inverse import save_inverse_mapping

    save_inverse_mapping(mapping, str(cache_path))
    log.info("biomech inverse: cached to %s", cache_path)
    return mapping


def _init(state: dict, config: dict) -> None:
    """Lazy-init: load the MyoSim3D model, the cached inverse mapping, and
    push the static index buffer to the bridge once."""
    from sigflow.nodes.app_display import _display_callback

    from sigflow.paths import resolve_path
    s3d_path = resolve_path(config["s3d_path"])
    cache_path = resolve_path(config["mapping_cache"])

    if not s3d_path.exists():
        log.error("biomech_tongue_display: .s3d model not found at %s", s3d_path)
        state["disabled_permanently"] = True
        return

    tongue = _build_tongue_model(s3d_path)
    mapping = _load_or_build_mapping(cache_path, s3d_path)

    state["tongue_model"] = tongue
    state["mapping"] = mapping
    state["indices_sent"] = False
    state["cal_frames"] = []
    state["cal_ref_model_yz"] = None  # set when calibration finalizes
    state["R"] = None
    state["t"] = None
    state["scale"] = 1.0
    state["_prev_phase"] = None

    # Push the static triangle index buffer once (same signature as
    # tongue_model_display — the bridge routes any "mesh_indices" to the
    # shared TongueMeshGeometry).
    if _display_callback is not None:
        index_bytes = tongue.to_index_buffer()
        _display_callback(
            config["display_id"],
            "mesh_indices",
            {"data": index_bytes, "has_uvs": False},
        )
        state["indices_sent"] = True
        log.info(
            "biomech tongue: %d atoms, %d triangles, mapping W=%s",
            len(tongue.positions),
            len(tongue.triangles()),
            mapping.W.shape,
        )

        # Push a rest-pose vertex buffer so the mesh is visible before tracking starts.
        tongue.set_activations(np.zeros(len(tongue.activations), dtype=np.float64))
        tongue.solve(max_iter=config.get("max_iter", 100))
        rest_buf = tongue.to_vertex_buffer()
        _display_callback(config["display_id"], "mesh", rest_buf)
        log.info("biomech tongue: rest pose emitted (%d bytes)", len(rest_buf))


def _finalize_calibration(state: dict) -> None:
    """Convert accumulated DLC calibration frames into a Procrustes-aligned
    reference midline in model (Y, Z) space, then cache the alignment so
    tracking frames can be projected the same way."""
    frames = state["cal_frames"]
    if not frames:
        log.warning("biomech_tongue_display: calibration ended with zero frames")
        return
    cal_mean_mm = np.mean(np.stack(frames, axis=0), axis=0)  # (11, 2)
    cal_model_yz = _us_to_model_yz(cal_mean_mm)               # (11, 2)

    mapping = state["mapping"]
    rest_ref = mapping.rest_keypoints  # (11, 2) in model (Y, Z)

    # Align the DLC calibration midline to the mapping's trained rest pose
    # so every tracking displacement lives in the same frame the ridge saw.
    R, t, scale = _procrustes(cal_model_yz, rest_ref)
    state["R"] = R
    state["t"] = t
    state["scale"] = scale
    state["cal_ref_model_yz"] = rest_ref.copy()
    log.info(
        "biomech tongue: calibration finalized on %d frames, scale=%.3f, "
        "translation=%s",
        len(frames),
        scale,
        t.round(2).tolist(),
    )


def _track(state: dict, config: dict, dorsal_mm: np.ndarray, lsl_timestamp: float) -> None:
    """Run a single tracking frame: ridge predict → forward solve → mesh update."""
    from sigflow.nodes.app_display import _display_callback

    if state["cal_ref_model_yz"] is None:
        # No calibration snapshot yet — bootstrap with current frame.
        state["cal_frames"].append(dorsal_mm.copy())
        if len(state["cal_frames"]) >= config.get("calibration_min_frames", 30):
            _finalize_calibration(state)
        return

    mapping = state["mapping"]
    tongue = state["tongue_model"]

    # Arc-length resample so the ridge feature vector lives on the same
    # parameterization it was trained on.
    mm_rs = _spline_arc_resample(dorsal_mm, n_out=11)

    # Map to model (Y, Z) and Procrustes-align to the trained rest frame.
    yz = _us_to_model_yz(mm_rs)
    aligned = state["scale"] * (yz @ state["R"]) + state["t"]

    # Displacement from trained rest midline → 22-dim flat feature.
    disp = (aligned - state["cal_ref_model_yz"]).reshape(-1)

    # Ridge inference (~2 µs on CPU).
    from sigflow.biomech.inverse import predict_activations

    pcts = predict_activations(disp, mapping.W, mapping.b)

    # Forward solve (~60 ms CPU NumPy, plenty fast for 13 Hz US).
    tongue.set_activations(pcts)
    tongue.solve(max_iter=int(config.get("max_iter", 100)))

    # Interleaved pos+normal buffer, 24-byte stride.
    buf = tongue.to_vertex_buffer()
    if _display_callback is not None:
        _display_callback(config["display_id"], "mesh", buf)

    state["last_activations"] = pcts
    state["last_ts"] = lsl_timestamp


# ─── Node entrypoint ────────────────────────────────────────────────


@sink_node(
    name="biomech_tongue_display",
    inputs=[Port("keypoints", Keypoints)],
    category="visualization",
    params=[
        Param("display_id", "str", "tongue_model", label="Display Target"),
        Param("enabled", "bool", False, label="Enable Biomech Rendering"),
        Param(
            "s3d_path",
            "str",
            "vendor/biomechanical-modelling/run/Tongue Model A – The price range.s3d",
            label="MyoSim3D .s3d Path",
        ),
        Param(
            "mapping_cache",
            "str",
            "weights/biomech_inverse_mapping.npz",
            label="Inverse Mapping Cache",
        ),
        Param("phase", "str", "calibration", label="Current Phase"),
        Param("calibration_min_frames", "int", 30, label="Min Calibration Frames"),
        Param("confidence_threshold", "float", 0.1, label="Min Keypoint Confidence"),
        Param("max_iter", "int", 100, label="Solver Iterations"),
    ],
)
def biomech_tongue_display(item, *, state, config):
    """Sigflow sink: DLC keypoints → muscle activations → mesh."""
    if not config.get("enabled", False):
        return
    if state.get("disabled_permanently"):
        return
    if "tongue_model" not in state:
        _init(state, config)
        if state.get("disabled_permanently"):
            return

    keypoints = item.data  # (16, 3) [x, y, confidence]
    if keypoints is None or len(keypoints) < 11:
        return

    dorsal_kp = keypoints[:11]
    confidence = dorsal_kp[:, 2]
    if float(confidence.mean()) < config.get("confidence_threshold", 0.1):
        return

    mm_per_pixel = item.metadata.get("mm_per_pixel", 1.0)
    if isinstance(mm_per_pixel, (list, tuple)):
        mm_per_pixel = mm_per_pixel[0]

    dorsal_mm = dorsal_kp[:, :2] * float(mm_per_pixel)  # (11, 2)

    phase = config.get("phase", "calibration")
    prev_phase = state.get("_prev_phase")
    if prev_phase is None:
        prev_phase = phase
    if phase != prev_phase and prev_phase == "calibration":
        _finalize_calibration(state)
    state["_prev_phase"] = phase

    if phase == "calibration":
        state["cal_frames"].append(dorsal_mm.copy())
        n = len(state["cal_frames"])
        if n % 10 == 0:
            log.debug("biomech calibration: %d frames collected", n)
        return

    _track(
        state,
        config,
        dorsal_mm,
        lsl_timestamp=getattr(item, "lsl_timestamp", 0.0),
    )
