"""Qt Quick 3D GPU-skinning tongue display.

The earlier implementation extracted **2D rotations** from segment-pair
geometry in the ultrasound (x, y) plane and packed them as quaternions
``(cos(θ/2), sin(θ/2), 0, 0)`` — a rotation purely about the model's X
axis. With Y and Z components forced to zero the chain could only flex in
a single plane, which is the cause of the "scrunched" geometry the user
observed.

Current implementation:

1. Build full 3D target positions for the dorsal joints, mapping
   ultrasound delta into model space (US x → model Z, US y → −model Y).
2. Run the shared anatomical-target pipeline
   (``sigflow.nodes.tongue_targets.compute_anatomical_targets``) — same
   confidence-weighted blend, per-bone stiffness ramp, arc-length
   conservation as the NumPy LBS path. This guarantees the two skinning
   paths consume identical anatomical targets.
3. Apply temporal smoothing on **target positions** (one-euro filter)
   rather than on rotations after the fact — smoothing positions before
   rotation derivation is more stable than smoothing rotations.
4. Run connected-chain FK (``compute_chain_fk``) to get bone world
   transforms with rest-pose bone lengths preserved.
5. For each joint, compute the local-space rotation quaternion as
   ``R_rest⁻¹ · R_world`` (``world_to_local_quaternion``), which is what
   Qt Quick 3D's ``Skin`` component expects.
"""
from __future__ import annotations

import logging

import numpy as np

from sigflow.node import Param, sink_node
from sigflow.nodes.tongue_targets import (
    AnatomicalTargetParams,
    compute_anatomical_targets,
    compute_chain_fk,
    world_to_local_quaternion,
)
from sigflow.types import Keypoints, Port

log = logging.getLogger(__name__)


def _one_euro_positions(state, key, x, t, min_cutoff=1.0, beta=0.007):
    """One-euro filter on a flat (n*3,) target buffer.

    Mirrors the LBS path's filter so smoothing kinetics are identical
    between the two skinning paths.
    """
    if key not in state:
        state[key] = (x.copy(), np.zeros_like(x), t)
        return x.copy()
    x_prev, dx_prev, t_prev = state[key]
    t_e = t - t_prev
    if t_e <= 0:
        return x_prev.copy()
    a_d = (2 * np.pi * 1.0 * t_e) / ((2 * np.pi * 1.0 * t_e) + 1.0)
    dx = (x - x_prev) / t_e
    dx_hat = a_d * dx + (1 - a_d) * dx_prev
    cutoff = min_cutoff + beta * np.abs(dx_hat)
    a = (2 * np.pi * cutoff * t_e) / ((2 * np.pi * cutoff * t_e) + 1.0)
    x_hat = a * x + (1 - a) * x_prev
    state[key] = (x_hat.copy(), dx_hat.copy(), t)
    return x_hat


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
    # Hard assertion — silent dtype demotions to U16 cause every vertex to
    # bind to bone 0 and the mesh moves as a single rigid body.
    assert buf["joints"].dtype == np.int32, (
        f"GLB joints buffer must be int32 for Qt Quick 3D Skin, got {buf['joints'].dtype}"
    )
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

    # Cache rest-pose joint positions and bone lengths in MM space so the
    # GPU path computes target positions on the same coordinate basis as
    # the LBS path (single source of truth for chain geometry).
    bone_rest_world_mm = mesh["bone_rest_world"].copy()
    bone_rest_world_mm[:, :3, 3] *= mm_per_model_unit
    state["bone_rest_world"] = bone_rest_world_mm.astype(np.float32)
    state["dorsal_rest_positions"] = bone_rest_world_mm[:11, :3, 3].copy().astype(np.float32)
    state["rest_bone_lengths"] = np.linalg.norm(
        np.diff(state["dorsal_rest_positions"], axis=0), axis=1,
    ).astype(np.float32)
    state["num_joints"] = mesh["num_joints"]

    _init_jaw_meshes(config, mesh, mm_per_model_unit)

    state["initialized"] = True
    state["cal_frames"] = []
    state["rest_kp_mm_2d"] = None      # mean of cal frames in US-mm 2D
    state["_prev_phase"] = None


def _finalize_calibration(state: dict) -> None:
    if state["cal_frames"]:
        # Calibration ref is in 2D US-mm space (the same space DLC outputs).
        # 3D model-space rest is fixed by the GLB and lives in
        # ``dorsal_rest_positions``.
        state["rest_kp_mm_2d"] = np.mean(state["cal_frames"], axis=0)


def _emit_joints(state: dict, config: dict, kp_mm_2d: np.ndarray, conf: np.ndarray, t: float) -> None:
    """Drive the 19 dorsal+ventral bones from the latest DLC keypoints.

    Pipeline mirrors the NumPy LBS path so the two skinning routes
    cannot disagree on what an "anatomically valid" deformation is.
    """
    rest_kp_mm_2d = state.get("rest_kp_mm_2d")
    if rest_kp_mm_2d is None:
        return

    rest_pos = state["dorsal_rest_positions"]
    rest_bone_lengths = state["rest_bone_lengths"]
    bone_rest_world = state["bone_rest_world"]
    num_joints = state["num_joints"]

    # Build raw 3D DLC targets (US x → model Z, US y → −model Y).
    delta_mm = kp_mm_2d - rest_kp_mm_2d                         # (11, 2)
    dlc_targets = rest_pos.copy()
    dlc_targets[:, 2] += delta_mm[:, 0].astype(np.float32)
    dlc_targets[:, 1] += -delta_mm[:, 1].astype(np.float32)

    # Anatomical-target pipeline (shared with LBS path).
    params = AnatomicalTargetParams(
        confidence_threshold=config.get("confidence_threshold", 0.1),
        confidence_soft_range=config.get("confidence_soft_range", 0.4),
        stiffness_root=config.get("stiffness_root", 0.5),
        stiffness_tip=config.get("stiffness_tip", 0.1),
        max_displacement_root_mm=config.get("max_displacement_root_mm", 8.0),
        max_displacement_tip_mm=config.get("max_displacement_tip_mm", 25.0),
        arc_length_min_ratio=config.get("arc_length_min_ratio", 0.92),
        arc_length_max_ratio=config.get("arc_length_max_ratio", 1.08),
    )
    targets = compute_anatomical_targets(
        dlc_targets=dlc_targets,
        rest_positions=rest_pos,
        confidences=conf,
        rest_bone_lengths=rest_bone_lengths,
        params=params,
    )

    # Temporal smoothing on positions (one-euro). Smoothing positions
    # before deriving rotations is more stable than SLERP-EMA on rotations
    # post-hoc.
    if t > 0:
        flat = targets.ravel()
        flat = _one_euro_positions(
            state, "_target_smoother", flat, t,
            min_cutoff=config.get("smooth_min_cutoff", 1.0),
            beta=config.get("smooth_beta", 0.007),
        )
        targets = flat.reshape(targets.shape).astype(np.float32)

    # Connected-chain FK with preserved rest-pose bone lengths.
    bone_world = compute_chain_fk(
        targets, bone_rest_world, rest_bone_lengths, num_dorsal=11,
    )

    # Convert each bone's world transform to a *local* quaternion
    # (R_rest_inv · R_world). Qt Quick 3D's Skin expects local rotations.
    # Dorsal joints 0–10 are articulated; ventral joints 11–18 stay at
    # rest, which means identity quaternion.
    quats: list[tuple[float, float, float, float]] = []
    for i in range(min(11, num_joints)):
        q = world_to_local_quaternion(bone_world[i], bone_rest_world[i])
        quats.append(q)
    for _ in range(11, num_joints):
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
        Param("confidence_soft_range", "float", 0.4,                     label="Confidence Soft Range"),
        Param("stiffness_root",        "float", 0.5,                     label="Stiffness (root)"),
        Param("stiffness_tip",         "float", 0.1,                     label="Stiffness (tip)"),
        Param("max_displacement_root_mm", "float", 8.0,                  label="Max Disp Root (mm)"),
        Param("max_displacement_tip_mm",  "float", 25.0,                 label="Max Disp Tip (mm)"),
        Param("arc_length_min_ratio", "float", 0.92,                     label="Arc-Length Min Ratio"),
        Param("arc_length_max_ratio", "float", 1.08,                     label="Arc-Length Max Ratio"),
        Param("smooth_min_cutoff",    "float", 1.0,                      label="Smoothing (lower=smoother)"),
        Param("smooth_beta",          "float", 0.007,                    label="Smoothing Speed Adapt"),
        # Legacy SLERP-EMA alpha — kept so old YAML protocols parse but
        # ignored; the new pipeline filters positions, not rotations.
        Param("smooth_alpha",          "float", 0.4,                     label="(deprecated)"),
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
    # DLC outputs 11 dorsal keypoints vallecula (root, z-) → tongueTip2 (anterior, z+).
    # TongueBond.glb's 11 dorsal bones are also ordered root → tip (bone 0 at z≈-2.2,
    # bone 10 at z≈+4.0 in model units), so keypoint i drives bone i directly.
    dorsal = keypoints[:11]
    confidence = dorsal[:, 2].astype(np.float32)
    if float(confidence.mean()) < float(config.get("confidence_threshold", 0.1)):
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
        if n >= int(config.get("calibration_min_frames", 30)) and state["rest_kp_mm_2d"] is None:
            _finalize_calibration(state)
            log.info("skinned_tongue_display: calibration finalised on %d frames", n)
        return

    t = item.lsl_timestamp if hasattr(item, "lsl_timestamp") else 0.0
    _emit_joints(state, config, kp_mm, confidence, t)
