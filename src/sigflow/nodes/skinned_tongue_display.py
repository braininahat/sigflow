"""Qt Quick 3D GPU-skinning tongue display — analytical keypoint→bone rotations from a skinned GLB."""
from __future__ import annotations

import logging

import numpy as np

from sigflow.node import Param, sink_node
from sigflow.types import Keypoints, Port

log = logging.getLogger(__name__)


def _slerp(q1, q2, t):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    dot = w1 * w2 + x1 * x2 + y1 * y2 + z1 * z2
    if dot < 0:
        w2, x2, y2, z2 = -w2, -x2, -y2, -z2
        dot = -dot
    if dot > 0.9995:
        w = w1 + t * (w2 - w1)
        x = x1 + t * (x2 - x1)
        y = y1 + t * (y2 - y1)
        z = z1 + t * (z2 - z1)
        n = (w * w + x * x + y * y + z * z) ** 0.5
        return (w / n, x / n, y / n, z / n)
    omega = np.arccos(np.clip(dot, -1, 1))
    sin_omega = np.sin(omega)
    s1 = np.sin((1 - t) * omega) / sin_omega
    s2 = np.sin(t * omega) / sin_omega
    return (s1 * w1 + s2 * w2, s1 * x1 + s2 * x2, s1 * y1 + s2 * y2, s1 * z1 + s2 * z2)


def _init_jaw_meshes(config: dict, mesh: dict, mm_per_model_unit: float) -> None:
    from sigflow.nodes.app_display import _display_callback

    static_meshes = mesh.get("static_meshes", {})
    if not static_meshes:
        log.warning("skinned_tongue_display: no static meshes in GLB")
        return

    jaw_data = {}
    for name, sdata in static_meshes.items():
        verts = sdata["vertices"] * mm_per_model_unit
        normals = sdata["normals"]
        indices = sdata["indices"]
        uvs = sdata.get("uvs")
        n_verts = len(verts)
        has_uvs = uvs is not None
        if has_uvs:
            interleaved = np.empty((n_verts, 8), dtype=np.float32)
            interleaved[:, :3] = verts
            interleaved[:, 3:6] = normals
            interleaved[:, 6:] = uvs
        else:
            interleaved = np.empty((n_verts, 6), dtype=np.float32)
            interleaved[:, :3] = verts
            interleaved[:, 3:] = normals
        jaw_data[name] = (interleaved.tobytes(), indices.astype(np.uint32).tobytes(), has_uvs)
        log.info("skinned_tongue_display: static mesh '%s': %d vertices (uvs=%s)", name, n_verts, has_uvs)

    if _display_callback is not None:
        _display_callback(config["display_id"], "mesh_static", jaw_data)
        log.info("skinned_tongue_display: emitted %d jaw meshes", len(jaw_data))


def _init_geometry(state: dict, config: dict) -> None:
    from pathlib import Path as _Path

    from sigflow.nodes._glb_mesh import parse_glb
    from sigflow.nodes.app_display import _display_callback
    from sigflow.paths import resolve_path

    model_path = config.get("model_path", "assets/TongueBond.glb")
    if not _Path(model_path).is_absolute():
        model_path = str(resolve_path(model_path))

    if not _Path(model_path).exists():
        log.error("skinned_tongue_display: GLB not found at %s", model_path)
        state["disabled_permanently"] = True
        return

    mesh = parse_glb(model_path)

    # Compute mm-per-model-unit from dorsal bone Z span so tongue, jaw, and
    # inv_bind matrices are all expressed in the same mm coordinate space.
    tongue_length_mm = config.get("tongue_length_mm", 70.0)
    dorsal_z = mesh["bone_rest_world"][:11, 2, 3]
    bone_z_span = float(dorsal_z.max() - dorsal_z.min())
    mm_per_model_unit = tongue_length_mm / (bone_z_span + 1e-12)

    n = mesh["num_vertices"]
    # Stride-56 interleaved layout: pos(12) + normal(12) + joints(16, i32×4) + weights(16, f32×4).
    # Qt Quick 3D's JointSemantic only accepts I32Type or F32Type for vertex attributes —
    # U16 is index-only and Qt logs "Attributes cannot be uint16, only index data" then
    # drops the attribute, leaving every vertex bound to bone 0 (mesh moves as a solid unit).
    dt = np.dtype([
        ("pos",     np.float32, 3),
        ("normal",  np.float32, 3),
        ("joints",  np.int32,   4),
        ("weights", np.float32, 4),
    ])
    buf = np.zeros(n, dtype=dt)
    buf["pos"]     = (mesh["vertices"] * mm_per_model_unit).astype(np.float32)
    buf["normal"]  = mesh["normals"].astype(np.float32)
    buf["joints"]  = mesh["joint_indices"].astype(np.int32)
    buf["weights"] = mesh["joint_weights"].astype(np.float32)
    vertex_bytes = buf.tobytes()

    index_bytes = mesh["indices"].astype(np.uint32).tobytes()

    # inv_bind matrices were baked in model units — scale translation column to mm
    # (row-major 4×4: rows 0-2, column 3 hold the translation)
    inv_bind_matrices = []
    for i in range(mesh["num_joints"]):
        mat = mesh["inv_bind_matrices"][i].copy()   # (4,4) row-major
        mat[:3, 3] *= mm_per_model_unit              # translation → mm
        inv_bind_matrices.append(mat.flatten().tolist())

    jaws_only = bool(config.get("jaws_only", False))
    if _display_callback is not None and not jaws_only:
        _display_callback(config["display_id"], "mesh_skin", {
            "vertices": vertex_bytes,
            "indices": index_bytes,
            "inv_bind_matrices": inv_bind_matrices,
            "bone_parents": mesh["bone_parents"].tolist(),
            "num_joints": mesh["num_joints"],
        })

    _init_jaw_meshes(config, mesh, mm_per_model_unit)

    state["initialized"] = True
    state["cal_frames"] = []
    state["rest_kp"] = None
    state["prev_quats"] = [(1.0, 0.0, 0.0, 0.0)] * 19
    state["_prev_phase"] = None


def _finalize_calibration(state: dict) -> None:
    if state["cal_frames"]:
        state["rest_kp"] = np.mean(state["cal_frames"], axis=0)


def _emit_joints(state: dict, config: dict, kp_mm: np.ndarray) -> None:
    rest_kp = state.get("rest_kp")
    if rest_kp is None:
        return

    alpha = config.get("smooth_alpha", 0.4)
    quats = []
    for i in range(10):
        rd = rest_kp[i + 1] - rest_kp[i]
        cd = kp_mm[i + 1] - kp_mm[i]
        rd_norm = np.linalg.norm(rd)
        cd_norm = np.linalg.norm(cd)
        if rd_norm < 1e-6 or cd_norm < 1e-6:
            quats.append(state["prev_quats"][i])
            continue
        rd = rd / rd_norm
        cd = cd / cd_norm
        cross = float(rd[0] * cd[1] - rd[1] * cd[0])
        dot = float(np.dot(rd, cd))
        angle_rad = np.arctan2(cross, dot)
        half = angle_rad / 2.0
        w = float(np.cos(half))
        x = float(np.sin(half))
        q = (w, x, 0.0, 0.0)
        q = _slerp(state["prev_quats"][i], q, alpha)
        quats.append(q)
        state["prev_quats"][i] = q

    # Tip bone (10): copy last dorsal segment rotation (no keypoint for tip)
    tip_q = quats[9] if len(quats) >= 10 else state["prev_quats"][10]
    quats.append(tip_q)
    state["prev_quats"][10] = tip_q

    # Ventral chain (11-18): identity — no DLC keypoint data
    for i in range(11, 19):
        quats.append((1.0, 0.0, 0.0, 0.0))

    from sigflow.nodes.app_display import _display_callback
    if _display_callback:
        _display_callback(config["display_id"], "tongue_joints", quats)


@sink_node(
    name="skinned_tongue_display",
    inputs=[Port("keypoints", Keypoints)],
    category="visualization",
    params=[
        Param("display_id",            "str",   "tongue_model",          label="Display Target"),
        Param("model_path",            "str",   "assets/TongueBond.glb", label="GLB Model Path"),
        Param("phase",                 "str",   "calibration",           label="Current Phase"),
        Param("calibration_min_frames","int",   30,                      label="Min Calibration Frames"),
        Param("confidence_threshold",  "float", 0.1,                     label="Min Keypoint Confidence"),
        Param("smooth_alpha",          "float", 0.4,                     label="Slerp EMA Alpha"),
        Param("tongue_length_mm",      "float", 70.0,                    label="Tongue Length (mm)"),
        Param("jaws_only",             "bool",  False,                   label="Skip tongue mesh (emit jaws only)"),
    ],
)
def skinned_tongue_display(item, *, state, config):
    if state.get("disabled_permanently"):
        return
    if not state.get("initialized"):
        from sigflow.nodes.app_display import _display_callback
        if _display_callback is None:
            return
        _init_geometry(state, config)
        if state.get("disabled_permanently"):
            return

    # jaws_only: jaw meshes are emitted once at init; no per-frame work after that
    if config.get("jaws_only", False):
        return

    keypoints = item.data
    if keypoints is None or len(keypoints) < 11:
        return
    # Reverse dorsal ordering so keypoint 0 maps to bone 0 (root/posterior).
    # DLC outputs vallecula (root) → tongueTip2 (anterior), but the GLB's bone
    # chain is ordered tip → root in the GLTF skin joint list.  Without the
    # reversal, every per-segment rotation computed by _emit_joints drives the
    # wrong bone, producing accordion-scrunched tongue deformation once the
    # i32 JointSemantic fix (sigflow #6) made per-vertex skinning actually
    # apply bone weights.
    dorsal = keypoints[:11][::-1]
    if float(dorsal[:, 2].mean()) < float(config.get("confidence_threshold", 0.1)):
        return

    mm_per_pixel = item.metadata.get("mm_per_pixel", 1.0)
    if isinstance(mm_per_pixel, (list, tuple)):
        mm_per_pixel = float(mm_per_pixel[0])
    kp_mm = dorsal[:, :2] * float(mm_per_pixel)

    phase = config.get("phase", "calibration")
    prev_phase = state.get("_prev_phase", phase)

    # Finalize calibration on phase transition
    if phase != prev_phase and prev_phase == "calibration":
        _finalize_calibration(state)
    state["_prev_phase"] = phase

    if phase == "calibration":
        state["cal_frames"].append(kp_mm.copy())
        n = len(state["cal_frames"])
        if n >= int(config.get("calibration_min_frames", 30)) and state["rest_kp"] is None:
            _finalize_calibration(state)
            log.info("skinned_tongue_display: calibration finalised on %d frames", n)
        return

    _emit_joints(state, config, kp_mm)
