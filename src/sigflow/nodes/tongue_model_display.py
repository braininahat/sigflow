"""3D tongue model sink — drives a skinned GLB mesh from DLC keypoints.

Supports two phases:
  - **calibration**: accumulates keypoint frames + jaw opening while the
    participant holds a vacuum hold.  Emits rest-pose mesh each frame.
  - **tracking**: computes displacements from the calibration reference,
    maps to model space, applies jaw offset from face landmarks, temporal
    smoothing, anatomical constraints, FK, and LBS.

Receives two inputs:
  - `keypoints` (Keypoints) — DLC tongue predictions from ultrasound
  - `landmarks` (FaceLandmarks, optional) — MediaPipe face mesh for jaw opening

Dorsal chain (joints 0–10) maps 1:1 to DLC keypoints 0–10.
Ventral chain (joints 11–18) stays at rest pose.
"""
import logging
import tempfile
from pathlib import Path

import numpy as np

from sigflow.node import sink_node, Param
from sigflow.nodes.tongue_targets import (
    AnatomicalTargetParams,
    compute_anatomical_targets,
)
from sigflow.types import Port, Keypoints, FaceLandmarks

log = logging.getLogger(__name__)

# --- One-Euro Filter ---


def _smoothing_factor(t_e, cutoff):
    r = 2 * np.pi * cutoff * t_e
    return r / (r + 1.0)


def _one_euro_filter(state_key, state, x, t, min_cutoff, beta, d_cutoff):
    """One-euro filter applied per-element to an array.

    State is stored in state[state_key] as (x_prev, dx_prev, t_prev).
    x: (N,) current values, t: timestamp in seconds.
    Returns filtered (N,) array.
    """
    if state_key not in state:
        state[state_key] = (x.copy(), np.zeros_like(x), t)
        return x.copy()

    x_prev, dx_prev, t_prev = state[state_key]
    t_e = t - t_prev
    if t_e <= 0:
        return x_prev.copy()

    # Derivative
    a_d = _smoothing_factor(t_e, d_cutoff)
    dx = (x - x_prev) / t_e
    dx_hat = a_d * dx + (1 - a_d) * dx_prev

    # Adaptive cutoff
    cutoff = min_cutoff + beta * np.abs(dx_hat)
    a = _smoothing_factor(t_e, cutoff)
    x_hat = a * x + (1 - a) * x_prev

    state[state_key] = (x_hat.copy(), dx_hat.copy(), t)
    return x_hat


# --- Coordinate mapping helpers ---


def _build_look_along_y(position, direction):
    """Build a 4×4 transform with Y axis along direction, at position."""
    y = direction / (np.linalg.norm(direction) + 1e-12)
    # Pick a reference axis for cross product (prefer world-X)
    ref = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    if abs(np.dot(y, ref)) > 0.9:
        ref = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    z = np.cross(y, ref)
    z /= np.linalg.norm(z) + 1e-12
    x = np.cross(y, z)
    x /= np.linalg.norm(x) + 1e-12
    m = np.eye(4, dtype=np.float32)
    m[:3, 0] = x
    m[:3, 1] = y
    m[:3, 2] = z
    m[:3, 3] = position
    return m


def _rotation_to_euler_xyz(R):
    """Rotation matrix → intrinsic XYZ Euler angles in degrees."""
    sy = np.sqrt(R[0, 0]**2 + R[1, 0]**2)
    if sy > 1e-6:
        x = np.arctan2(R[2, 1], R[2, 2])
        y = np.arctan2(-R[2, 0], sy)
        z = np.arctan2(R[1, 0], R[0, 0])
    else:
        x = np.arctan2(-R[1, 2], R[1, 1])
        y = np.arctan2(-R[2, 0], sy)
        z = 0
    return np.degrees(np.array([x, y, z], dtype=np.float32))


def _smooth_targets_spline(targets, smoothing=0.5):
    """Smooth 11 dorsal target positions with a cubic spline.

    Parameterized by cumulative arc length. Smooths Y and Z coordinates
    independently (X stays constant for the midsagittal plane).
    """
    from scipy.interpolate import UnivariateSpline

    n = len(targets)
    if n < 4:
        return targets

    # Parameterize by cumulative arc length along target chain
    diffs = np.diff(targets, axis=0)
    seg_lengths = np.linalg.norm(diffs, axis=1)
    t = np.zeros(n, dtype=np.float64)
    t[1:] = np.cumsum(seg_lengths)
    if t[-1] > 0:
        t /= t[-1]

    smoothed = targets.copy()
    for axis in (1, 2):  # Y (vertical) and Z (anterior-posterior)
        try:
            spl = UnivariateSpline(t, targets[:, axis], k=3, s=smoothing)
            smoothed[:, axis] = spl(t)
        except Exception:
            pass  # fall back to unsmoothed on degenerate input
    return smoothed


_POSE_INDICES = [6, 10, 33, 133, 263, 362]  # nose bridge, forehead, eye corners


def _compute_head_rotation(ref_pts, cur_pts):
    """SVD Procrustes: rotation delta from reference to current landmark set."""
    ref_c = ref_pts - ref_pts.mean(axis=0)
    cur_c = cur_pts - cur_pts.mean(axis=0)
    H = ref_c.T @ cur_c
    U, _, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    return _rotation_to_euler_xyz(R)


def _collapse_ventral(deformed_verts, dorsalness, threshold=0.5):
    """Collapse ventral vertices to the dorsal centroid so their triangles vanish."""
    dorsal_mask = dorsalness >= threshold
    if dorsal_mask.any():
        centroid = deformed_verts[dorsal_mask].mean(axis=0)
        deformed_verts[~dorsal_mask] = centroid
    return deformed_verts


def _emit_rest_mesh(state, config, display_callback):
    """Emit the rest-pose tongue mesh (used during calibration)."""
    bone_transforms = state["bone_rest_world"]
    deformed_verts, deformed_normals = _apply_lbs(
        state["vertices"], state["normals"],
        state["joint_indices"], state["joint_weights"],
        bone_transforms, state["inv_bind_matrices"],
    )
    # Hide ventral surface
    if "vertex_dorsalness" in state:
        deformed_verts = _collapse_ventral(deformed_verts, state["vertex_dorsalness"])
    uvs = state.get("uvs")
    if uvs is not None:
        interleaved = np.empty((state["num_vertices"], 8), dtype=np.float32)
        interleaved[:, :3] = deformed_verts
        interleaved[:, 3:6] = deformed_normals
        interleaved[:, 6:] = uvs
    else:
        interleaved = np.empty((state["num_vertices"], 6), dtype=np.float32)
        interleaved[:, :3] = deformed_verts
        interleaved[:, 3:] = deformed_normals
    if display_callback:
        display_callback(config["display_id"], "mesh", interleaved.tobytes())


def _finalize_calibration(state):
    """Compute reference pose from accumulated calibration frames."""
    cal_frames = state.get("cal_frames", [])
    if not cal_frames:
        log.warning("finalize_calibration: no frames accumulated")
        return

    stacked = np.stack(cal_frames)                      # (N, 11, 2)
    state["ref_kp_mm"] = stacked.mean(axis=0)           # (11, 2)

    cal_jaw = state.get("cal_jaw_openings", [])
    state["ref_jaw_opening"] = float(np.mean(cal_jaw)) if cal_jaw else 0.0

    if state.get("cal_pose_pts"):
        state["ref_pose_pts"] = np.mean(state["cal_pose_pts"], axis=0)
    state["ref_mouth_open"] = float(np.mean(state["cal_mouth_open"])) if state.get("cal_mouth_open") else 0.0

    log.info("calibration finalized: %d frames, ref_jaw=%.4f, ref_mouth=%.4f",
             len(cal_frames), state["ref_jaw_opening"], state["ref_mouth_open"])


def _compute_bone_transforms(target_positions, rest_transforms, rest_bone_lengths, num_dorsal=11):
    """Compute world transforms via connected-chain FK with preserved bone lengths.

    Builds a connected joint chain: each joint's position is determined by the
    previous joint's position + direction toward the target × rest-pose bone
    length. This preserves bone lengths (no stretching/compression) while using
    keypoint predictions to guide joint directions.

    Args:
        target_positions: (11, 3) mm-space target positions for dorsal joints
        rest_transforms: (J, 4, 4) rest-pose world transforms
        rest_bone_lengths: (10,) rest-pose distances between consecutive dorsal joints
        num_dorsal: number of dorsal chain bones (default 11)

    Returns:
        (J, 4, 4) new world transforms
    """
    transforms = rest_transforms.copy()

    # Build connected joint chain with preserved bone lengths
    chain = np.empty((num_dorsal, 3), dtype=np.float32)
    chain[0] = target_positions[0]  # Root anchored at first target

    prev_direction = np.array([0.0, 1.0, 0.0], dtype=np.float32)

    for i in range(1, num_dorsal):
        # Direction from current chain position toward this joint's target
        direction = target_positions[i] - chain[i - 1]
        norm = np.linalg.norm(direction)
        if norm > 1e-6:
            direction /= norm
            prev_direction = direction
        else:
            direction = prev_direction

        # Advance by rest-pose bone length (preserves rigidity)
        chain[i] = chain[i - 1] + direction * rest_bone_lengths[i - 1]

    # Build transforms: origin at joint, Y-axis along direction to next joint
    for i in range(num_dorsal):
        if i < num_dorsal - 1:
            direction = chain[i + 1] - chain[i]
            norm = np.linalg.norm(direction)
            if norm > 1e-6:
                direction /= norm
            else:
                direction = prev_direction
        else:
            # Last bone: use direction from previous bone
            direction = chain[i] - chain[i - 1]
            norm = np.linalg.norm(direction)
            if norm > 1e-6:
                direction /= norm
            else:
                direction = prev_direction

        transforms[i] = _build_look_along_y(chain[i], direction)

    # Ventral chain (joints num_dorsal..num_joints-1) stays at rest pose
    return transforms


def _apply_lbs(vertices, normals, joint_indices, joint_weights, bone_transforms, inv_bind_matrices):
    """Apply linear blend skinning (vectorized numpy).

    Args:
        vertices: (V, 3) rest positions
        normals: (V, 3) rest normals
        joint_indices: (V, 4) uint8
        joint_weights: (V, 4) float32
        bone_transforms: (J, 4, 4) current world transforms
        inv_bind_matrices: (J, 4, 4) inverse bind matrices

    Returns:
        (deformed_vertices, deformed_normals) — each (V, 3) float32
    """
    V = vertices.shape[0]

    # Precompute skinning matrices: bone_world @ ibm for each joint
    skin_matrices = bone_transforms @ inv_bind_matrices  # (J, 4, 4)

    # Homogeneous vertices: (V, 4)
    verts_h = np.ones((V, 4), dtype=np.float32)
    verts_h[:, :3] = vertices

    # For each of the 4 bone influences, accumulate weighted transform
    deformed = np.zeros((V, 4), dtype=np.float32)
    deformed_n = np.zeros((V, 3), dtype=np.float32)

    for k in range(4):
        ji = joint_indices[:, k]  # (V,) joint index for influence k
        w = joint_weights[:, k:k+1]  # (V, 1) weight

        # Gather the 4×4 matrix for each vertex's k-th influence
        M = skin_matrices[ji]  # (V, 4, 4)

        # Transform positions: (V, 4, 4) @ (V, 4, 1) → (V, 4, 1)
        transformed = np.einsum('vij,vj->vi', M, verts_h)  # (V, 4)
        deformed += transformed * w

        # Transform normals (rotation only, upper-left 3×3)
        transformed_n = np.einsum('vij,vj->vi', M[:, :3, :3], normals)  # (V, 3)
        deformed_n += transformed_n * w

    # Normalize normals
    norms = np.linalg.norm(deformed_n, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    deformed_n /= norms

    return deformed[:, :3], deformed_n


def _compute_leveling_rotation(uj_verts):
    """Compute rotation that levels the upper jaw occlusal plane.

    Uses heightmap floor: bins upper jaw into XZ spatial bins, takes min-Y
    per bin (the tooth surface), fits a plane via SVD, then computes a
    Rodrigues rotation from the actual normal to [0, -1, 0].

    Returns (R3x3, R4x4) float32 rotation matrices.  Identity if already level.
    """
    BIN = 2.0
    x_lo, x_hi = float(uj_verts[:, 0].min()), float(uj_verts[:, 0].max())
    z_lo, z_hi = float(uj_verts[:, 2].min()), float(uj_verts[:, 2].max())
    nx = int((x_hi - x_lo) / BIN) + 1
    nz = int((z_hi - z_lo) / BIN) + 1
    bx = np.clip(((uj_verts[:, 0] - x_lo) / BIN).astype(np.intp), 0, nx - 1)
    bz = np.clip(((uj_verts[:, 2] - z_lo) / BIN).astype(np.intp), 0, nz - 1)
    floor_y = np.full((nx, nz), np.inf, dtype=np.float32)
    np.minimum.at(floor_y, (bx, bz), uj_verts[:, 1])
    valid = np.isfinite(floor_y)
    ix, iz = np.where(valid)
    cusp_pts = np.column_stack([
        x_lo + (ix + 0.5) * BIN,
        floor_y[valid],
        z_lo + (iz + 0.5) * BIN,
    ]).astype(np.float64)

    centroid = cusp_pts.mean(axis=0)
    _, _, Vt = np.linalg.svd(cusp_pts - centroid, full_matrices=False)
    normal = Vt[-1]
    if normal[1] > 0:
        normal = -normal

    target = np.array([0.0, -1.0, 0.0], dtype=np.float64)
    normal_n = normal / (np.linalg.norm(normal) + 1e-12)

    dot = np.clip(np.dot(normal_n, target), -1.0, 1.0)
    if abs(dot - 1.0) < 1e-8:
        return np.eye(3, dtype=np.float32), np.eye(4, dtype=np.float32)

    cross = np.cross(normal_n, target)
    sin_angle = np.linalg.norm(cross)

    if sin_angle < 1e-8:
        perp = np.array([1, 0, 0], dtype=np.float64)
        if abs(np.dot(normal_n, perp)) > 0.9:
            perp = np.array([0, 0, 1], dtype=np.float64)
        axis = np.cross(normal_n, perp)
        axis /= np.linalg.norm(axis)
        angle = np.pi
    else:
        axis = cross / sin_angle
        angle = np.arccos(dot)

    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0],
    ], dtype=np.float64)
    R3 = (np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)).astype(np.float32)

    R4 = np.eye(4, dtype=np.float32)
    R4[:3, :3] = R3
    log.info("occlusal leveling: %.2f° rotation", np.degrees(angle))
    return R3, R4


def _rotate_verts_x(verts, pivot, angle_deg):
    """Rotate vertices around X axis at pivot by angle_deg (in YZ plane)."""
    rad = np.radians(angle_deg)
    c, s = np.cos(rad), np.sin(rad)
    rel = verts - pivot
    out = verts.copy()
    out[:, 1] = rel[:, 1] * c - rel[:, 2] * s + pivot[1]
    out[:, 2] = rel[:, 1] * s + rel[:, 2] * c + pivot[2]
    return out


def _load_rest_pose(state, scaled_static):
    """Load saved rest pose from disk and apply to state and static jaw meshes.

    Overrides dorsal/ventral rest positions and bone_rest_world transforms.
    Keeps original inv_bind_matrices (GLB bind pose) so LBS deforms correctly.
    Applies saved jaw transforms (UJ offset/rot, LJ offset/open/scale) to static meshes.

    Returns True if a rest pose was loaded, False otherwise.
    """
    rest_path = Path("assets/tongue_rest_pose.npz")
    if not rest_path.exists():
        return False

    log.info("loading rest pose from %s", rest_path)
    saved = np.load(rest_path)

    # --- Dorsal chain ---
    dorsal_pos = saved["positions"].astype(np.float32)   # (11, 3)
    state["dorsal_rest_positions"] = dorsal_pos.copy()
    state["rest_bone_lengths"] = np.linalg.norm(np.diff(dorsal_pos, axis=0), axis=1)

    # Recompute bone_rest_world transforms for dorsal bones
    bone_rest = state["bone_rest_world"].copy()
    num_dorsal = len(dorsal_pos)
    for i in range(num_dorsal):
        if i < num_dorsal - 1:
            d = dorsal_pos[i + 1] - dorsal_pos[i]
        else:
            d = dorsal_pos[i] - dorsal_pos[i - 1]
        n = np.linalg.norm(d)
        d = d / n if n > 1e-6 else np.array([0, 1, 0], dtype=np.float32)
        bone_rest[i] = _build_look_along_y(dorsal_pos[i], d)

    # --- Ventral chain ---
    if "ventral_positions" in saved:
        ventral_pos = saved["ventral_positions"].astype(np.float32)  # (8, 3)
        nv = len(ventral_pos)
        for i in range(nv):
            j = num_dorsal + i
            if i < nv - 1:
                d = ventral_pos[i + 1] - ventral_pos[i]
            else:
                d = ventral_pos[i] - ventral_pos[i - 1]
            n = np.linalg.norm(d)
            d = d / n if n > 1e-6 else np.array([0, 1, 0], dtype=np.float32)
            bone_rest[j] = _build_look_along_y(ventral_pos[i], d)

    state["bone_rest_world"] = bone_rest
    # NOTE: inv_bind_matrices stays from original GLB bind pose — LBS needs this

    # --- Jaw transforms (applied to already-aligned static meshes) ---
    uj_offset = saved.get("uj_offset")
    uj_rot = float(saved.get("uj_rot", 0.0))
    lj_offset = saved.get("lj_offset")
    jaw_open = float(saved.get("jaw_open", 0.0))
    lj_y_scale = float(saved.get("lj_y_scale", 1.0))

    if "upper_jaw" in scaled_static:
        v = scaled_static["upper_jaw"]["vertices"]
        if uj_offset is not None:
            v = v + uj_offset.astype(np.float32)
        if abs(uj_rot) > 0.01:
            pivot = np.array([v[:, 0].mean(), v[:, 1].mean(), v[:, 2].min()],
                             dtype=np.float32)
            v = _rotate_verts_x(v, pivot, uj_rot)
        scaled_static["upper_jaw"]["vertices"] = v

    if "lower_jaw" in scaled_static:
        v = scaled_static["lower_jaw"]["vertices"]
        if abs(lj_y_scale - 1.0) > 1e-4:
            mean_y = v[:, 1].mean()
            v[:, 1] = mean_y + (v[:, 1] - mean_y) * lj_y_scale
        if lj_offset is not None:
            v = v + lj_offset.astype(np.float32)
        scaled_static["lower_jaw"]["vertices"] = v

    state["jaw_close_offset"] = jaw_open

    # The articulator's preview rotates the LJ around `tmj_base + uj_offset`,
    # which can't be reconstructed from the post-transform UJ/LJ vertices that
    # the runtime sees here (uj_rot in particular changes the UJ y-min that
    # the auto-TMJ formula depends on).  When the articulator saved a TMJ
    # explicitly we use it verbatim so the rotation pivot matches the preview.
    if "tmj_position" in saved.files:
        state["tmj_position"] = saved["tmj_position"].astype(np.float32)

    log.info("rest pose loaded: dorsal %s, uj_rot=%.1f°, jaw_open=%.1f°, lj_scale=%.2f",
             dorsal_pos.shape, uj_rot, jaw_open, lj_y_scale)
    return True


def _compute_jaw_y_offset(uj_verts, lj_verts):
    """Compute Y translation that aligns the lower jaw molars with the upper jaw.

    Uses heightmap binning to find the per-XZ-bin gap between the upper jaw
    floor (lowest Y) and lower jaw ceiling (highest Y).  Returns the 5th
    percentile gap minus a 1mm margin so the tightest contact points just touch.
    """
    BIN = 2.0  # mm per spatial bin

    # Overlapping XZ region only
    x_lo = max(float(uj_verts[:, 0].min()), float(lj_verts[:, 0].min()))
    x_hi = min(float(uj_verts[:, 0].max()), float(lj_verts[:, 0].max()))
    z_lo = max(float(uj_verts[:, 2].min()), float(lj_verts[:, 2].min()))
    z_hi = min(float(uj_verts[:, 2].max()), float(lj_verts[:, 2].max()))

    nx = int((x_hi - x_lo) / BIN) + 1
    nz = int((z_hi - z_lo) / BIN) + 1

    def _bin(v, lo, n):
        return np.clip(((v - lo) / BIN).astype(np.intp), 0, n - 1)

    # Upper jaw floor: minimum Y per XZ bin
    uj_in = uj_verts[(uj_verts[:, 0] >= x_lo) & (uj_verts[:, 0] <= x_hi) &
                      (uj_verts[:, 2] >= z_lo) & (uj_verts[:, 2] <= z_hi)]
    uj_floor = np.full((nx, nz), np.inf, dtype=np.float32)
    np.minimum.at(uj_floor,
                  (_bin(uj_in[:, 0], x_lo, nx), _bin(uj_in[:, 2], z_lo, nz)),
                  uj_in[:, 1])

    # Lower jaw ceiling: maximum Y per XZ bin
    lj_in = lj_verts[(lj_verts[:, 0] >= x_lo) & (lj_verts[:, 0] <= x_hi) &
                      (lj_verts[:, 2] >= z_lo) & (lj_verts[:, 2] <= z_hi)]
    lj_ceil = np.full((nx, nz), -np.inf, dtype=np.float32)
    np.maximum.at(lj_ceil,
                  (_bin(lj_in[:, 0], x_lo, nx), _bin(lj_in[:, 2], z_lo, nz)),
                  lj_in[:, 1])

    both = np.isfinite(uj_floor) & np.isfinite(lj_ceil)
    gaps = uj_floor[both] - lj_ceil[both]

    # 5th percentile gap - 1mm margin → tightest molars just touch
    return float(np.percentile(gaps, 5)) - 1.0


def _init_mesh(state, config):
    """Parse GLB and cache mesh data + rest-pose bone info.

    Rescales everything from model units to mm using the reference tongue length.
    Sends static jaw mesh data once via display callback.
    """
    from sigflow.nodes._glb_mesh import parse_glb
    from sigflow.nodes.app_display import _display_callback

    model_path = config["model_path"]
    if not Path(model_path).is_absolute():
        model_path = str(Path(model_path).resolve())

    log.info("loading tongue model from %s", model_path)
    mesh = parse_glb(model_path)

    # --- mm-space rescaling ---
    # Dorsal bone chain Z span in model units
    dorsal_z = mesh["bone_rest_world"][:11, 2, 3]
    bone_z_span = dorsal_z.max() - dorsal_z.min()
    tongue_length_mm = config.get("tongue_length_mm", 70.0)
    mm_per_model_unit = tongue_length_mm / (bone_z_span + 1e-12)

    log.info("model Z span=%.3f units, tongue_length=%.1f mm, scale=%.3f mm/unit",
             bone_z_span, tongue_length_mm, mm_per_model_unit)

    # Scale vertex positions
    vertices = mesh["vertices"] * mm_per_model_unit
    normals = mesh["normals"]  # normals are direction-only, no scale

    # Scale bone rest world translations
    bone_rest_world = mesh["bone_rest_world"].copy()
    bone_rest_world[:, :3, 3] *= mm_per_model_unit

    # Store UVs (None if not present in GLB)
    state["uvs"] = mesh.get("uvs")

    # --- Occlusal plane leveling ---
    # Scale static jaw meshes first (needed for leveling computation)
    static_meshes = mesh.get("static_meshes", {})
    scaled_static = {}
    for name, smesh in static_meshes.items():
        scaled_static[name] = {
            "vertices": smesh["vertices"] * mm_per_model_unit,
            "normals": smesh["normals"],
            "indices": smesh["indices"],
            "uvs": smesh.get("uvs"),
            "material_idx": smesh.get("material_idx"),
        }

    if "upper_jaw" in scaled_static:
        R3_level, R4_level = _compute_leveling_rotation(scaled_static["upper_jaw"]["vertices"])
        # Apply leveling to all vertex data
        vertices = (R3_level @ vertices.T).T
        normals = (R3_level @ normals.T).T
        for name in scaled_static:
            scaled_static[name]["vertices"] = (R3_level @ scaled_static[name]["vertices"].T).T
            scaled_static[name]["normals"] = (R3_level @ scaled_static[name]["normals"].T).T
        # Apply leveling to bone_rest_world: R4 @ each bone transform
        bone_rest_world = np.einsum('ij,njk->nik', R4_level, bone_rest_world)

    # Recompute inverse bind matrices from leveled+scaled rest world
    inv_bind_matrices = np.linalg.inv(bone_rest_world)

    state["vertices"] = vertices
    state["normals"] = normals
    state["indices"] = mesh["indices"]
    state["joint_indices"] = mesh["joint_indices"]
    state["joint_weights"] = mesh["joint_weights"]
    state["inv_bind_matrices"] = inv_bind_matrices
    state["bone_rest_world"] = bone_rest_world
    state["num_joints"] = mesh["num_joints"]
    state["num_vertices"] = mesh["num_vertices"]

    # Cache dorsal rest positions (now in mm) and bone lengths
    state["dorsal_rest_positions"] = bone_rest_world[:11, :3, 3].copy()
    dorsal_pos = state["dorsal_rest_positions"]
    state["rest_bone_lengths"] = np.linalg.norm(
        np.diff(dorsal_pos, axis=0), axis=1
    )  # (10,)

    # Per-vertex dorsalness: fraction of skin weight on dorsal joints (0–10)
    # Used to hide ventral surface (only show dorsal tongue)
    dorsalness = np.zeros(mesh["num_vertices"], dtype=np.float32)
    for k in range(4):
        is_dorsal = mesh["joint_indices"][:, k] < 11
        dorsalness += mesh["joint_weights"][:, k] * is_dorsal
    state["vertex_dorsalness"] = dorsalness

    # Precompute index buffer bytes (static, sent once)
    state["index_bytes"] = mesh["indices"].astype(np.uint32).tobytes()
    state["indices_sent"] = False

    # Align lower jaw: translate Y so molar surfaces meet upper jaw
    display_id = config["display_id"]
    if "upper_jaw" in scaled_static and "lower_jaw" in scaled_static:
        y_offset = _compute_jaw_y_offset(
            scaled_static["upper_jaw"]["vertices"],
            scaled_static["lower_jaw"]["vertices"],
        )
        scaled_static["lower_jaw"]["vertices"] = (
            scaled_static["lower_jaw"]["vertices"] + np.array([0, y_offset, 0], dtype=np.float32)
        )
        log.info("jaw Y-alignment offset=%.1f mm", y_offset)

    # Load saved rest pose (overrides dorsal/ventral positions + jaw transforms)
    _load_rest_pose(state, scaled_static)

    # Send static jaw meshes (now aligned + rest-pose adjusted) once
    if scaled_static and _display_callback:
        jaw_data = {}
        for name, sdata in scaled_static.items():
            n_verts = len(sdata["vertices"])
            has_uvs = sdata.get("uvs") is not None
            if has_uvs:
                interleaved = np.empty((n_verts, 8), dtype=np.float32)
                interleaved[:, :3] = sdata["vertices"]
                interleaved[:, 3:6] = sdata["normals"]
                interleaved[:, 6:] = sdata["uvs"]
            else:
                interleaved = np.empty((n_verts, 6), dtype=np.float32)
                interleaved[:, :3] = sdata["vertices"]
                interleaved[:, 3:] = sdata["normals"]
            jaw_data[name] = (interleaved.tobytes(), sdata["indices"].astype(np.uint32).tobytes(), has_uvs)
            log.info("static mesh '%s': %d vertices (uvs=%s)", name, n_verts, has_uvs)
        _display_callback(display_id, "mesh_static", jaw_data)

    # Extract and send texture images + material properties
    glb_textures = mesh.get("textures", {})
    materials_info = mesh.get("materials_info", {})
    if _display_callback and (glb_textures or materials_info):
        texture_paths = {}
        material_props = {}
        # Map mesh name → material index
        mat_map = {"tongue": mesh.get("skinned_material_idx")}
        for name, sdata in scaled_static.items():
            mat_map[name] = sdata.get("material_idx")
        for mesh_name, mat_idx in mat_map.items():
            if mat_idx is None:
                continue
            # Save texture images to temp files
            if mat_idx in glb_textures:
                for tex_type, tex in glb_textures[mat_idx].items():
                    ext = ".png" if "png" in tex["mime"] else ".jpg"
                    tmp = tempfile.NamedTemporaryFile(
                        suffix=ext, prefix=f"{mesh_name}_{tex_type}_", delete=False)
                    tmp.write(tex["data"])
                    tmp.close()
                    texture_paths.setdefault(mesh_name, {})[tex_type] = tmp.name
                    log.info("texture '%s/%s': %s (%d bytes)",
                             mesh_name, tex_type, tmp.name, len(tex["data"]))
            # Collect material properties
            if mat_idx in materials_info:
                material_props[mesh_name] = materials_info[mat_idx]
        if texture_paths or material_props:
            _display_callback(display_id, "textures",
                              {"paths": texture_paths, "materials": material_props})

    # Detect TMJ pivot: above and behind the lower jaw's posterior edge.
    # Anatomically the TMJ sits at ear level, behind the last molar.
    # _load_rest_pose may have already populated tmj_position from the
    # articulator's saved npz — in that case the saved value wins because
    # it matches the rotation pivot the user actually saw in the preview.
    if "tmj_position" not in state:
        if "lower_jaw" in scaled_static:
            jaw_verts = scaled_static["lower_jaw"]["vertices"]
            tmj_x = jaw_verts[:, 0].mean()
            tmj_z = jaw_verts[:, 2].min() - 5.0
            if "upper_jaw" in scaled_static:
                tmj_y = scaled_static["upper_jaw"]["vertices"][:, 1].min() + 5.0
            else:
                tmj_y = jaw_verts[:, 1].max() + 5.0
            state["tmj_position"] = np.array([tmj_x, tmj_y, tmj_z], dtype=np.float32)
        else:
            state["tmj_position"] = state["dorsal_rest_positions"][0] + np.array([0, 15, -5], dtype=np.float32)
    if _display_callback:
        _display_callback(display_id, "tmj_position", state["tmj_position"])

    # jaw_close_offset is set by _load_rest_pose if a saved pose exists
    # (= the articulator's saved jaw_open).  Default to 0 (neutral) only if
    # no saved pose was found, so the saved closure isn't silently clobbered.
    state.setdefault("jaw_close_offset", 0.0)

    log.info("tongue model loaded: %d vertices, %d joints (mm-space), tmj=%s, jaw_close=%.1f°",
             mesh["num_vertices"], mesh["num_joints"], state["tmj_position"],
             state["jaw_close_offset"])


@sink_node(
    name="tongue_model_display",
    inputs=[Port("keypoints", Keypoints), Port("landmarks", FaceLandmarks)],
    category="visualization",
    params=[
        Param("display_id", "str", "tongue_model", label="Display Target"),
        Param("model_path", "str", "assets/TongueBond.glb", label="Model Path"),
        Param("confidence_threshold", "float", 0.1, label="Min Confidence"),
        # Confidence-weighted target blend (tongue_targets.py)
        Param("confidence_soft_range", "float", 0.4, label="Confidence Soft Range"),
        # Per-bone stiffness / displacement ramps. Root values are stiffer
        # than tip — anatomically the tongue body is anchored, the tip free.
        Param("stiffness_root", "float", 0.5, label="Stiffness (root)"),
        Param("stiffness_tip", "float", 0.1, label="Stiffness (tip)"),
        Param("max_displacement_root_mm", "float", 8.0, label="Max Disp Root (mm)"),
        Param("max_displacement_tip_mm", "float", 25.0, label="Max Disp Tip (mm)"),
        # Arc-length / muscular-hydrostat conservation
        Param("arc_length_min_ratio", "float", 0.92, label="Arc-Length Min Ratio"),
        Param("arc_length_max_ratio", "float", 1.08, label="Arc-Length Max Ratio"),
        # Legacy uniform stiffness — kept so existing YAML protocols keep
        # parsing; superseded by the per-bone ramp above and ignored.
        Param("stiffness", "float", 0.1, label="(deprecated)"),
        Param("max_displacement_mm", "float", 25.0, label="(deprecated)"),
        Param("tongue_length_mm", "float", 70.0, label="Reference Tongue Length (mm)"),
        Param("smooth_min_cutoff", "float", 1.0, label="Smoothing (lower=smoother)"),
        Param("smooth_beta", "float", 0.007, label="Smoothing Speed Adapt"),
        Param("phase", "str", "calibration", label="Current Phase"),
        Param("calibration_min_frames", "int", 30, label="Min Calibration Frames"),
        Param("mandible_angle_scale", "float", 100.0, label="Mandible Angle Scale"),
        Param("tmj_coupled_factor", "float", 0.3, label="TMJ Coupled Factor (mm/deg)"),
        Param("spline_smoothing", "float", 0.5, label="Spatial Spline Smoothing"),
    ],
)
def tongue_model_display(item, *, state, config):
    from sigflow.nodes.app_display import _display_callback

    if "vertices" not in state:
        _init_mesh(state, config)

    display_id = config["display_id"]

    # Send static index buffer once
    if not state["indices_sent"] and _display_callback:
        has_uvs = state.get("uvs") is not None
        _display_callback(display_id, "mesh_indices", {"data": state["index_bytes"], "has_uvs": has_uvs})
        state["indices_sent"] = True

    # --- Multi-input routing ---
    if issubclass(item.port_type, FaceLandmarks):
        lm = item.data  # (468, 3)
        state["face_pose_pts"] = lm[_POSE_INDICES]

        upper_lip, lower_lip = lm[13], lm[14]
        face_top, chin = lm[10], lm[152]
        face_height = abs(chin[1] - face_top[1]) + 1e-6
        state["mouth_open_norm"] = np.linalg.norm(lower_lip[:2] - upper_lip[:2]) / face_height
        return

    # --- Keypoints processing ---
    keypoints = item.data  # (16, 3) → [x, y, confidence]
    if keypoints is None or len(keypoints) < 11:
        return

    dorsal_kp = keypoints[:11]
    confidence = dorsal_kp[:, 2]
    threshold = config["confidence_threshold"]

    valid = confidence >= threshold
    if valid.sum() < 3:
        return

    mm_per_pixel = item.metadata.get("mm_per_pixel", 1.0)
    if isinstance(mm_per_pixel, (list, tuple)):
        mm_per_pixel = mm_per_pixel[0]

    # Interpolate invalid keypoints from neighbors
    kp_px = dorsal_kp[:, :2].copy()
    if not valid.all():
        valid_indices = np.where(valid)[0]
        invalid_indices = np.where(~valid)[0]
        for idx in invalid_indices:
            left = valid_indices[valid_indices < idx]
            right = valid_indices[valid_indices > idx]
            if len(left) > 0 and len(right) > 0:
                l, r = left[-1], right[0]
                t_interp = (idx - l) / (r - l)
                kp_px[idx] = kp_px[l] * (1 - t_interp) + kp_px[r] * t_interp
            elif len(left) > 0:
                kp_px[idx] = kp_px[left[-1]]
            elif len(right) > 0:
                kp_px[idx] = kp_px[right[0]]

    t = item.lsl_timestamp if hasattr(item, "lsl_timestamp") else 0.0

    # Detect phase transition
    phase = config.get("phase", "calibration")
    prev_phase = state.get("_prev_phase")
    if prev_phase is None:
        prev_phase = phase
    if phase != prev_phase and prev_phase == "calibration":
        _finalize_calibration(state)
    state["_prev_phase"] = phase

    # Convert to raw mm (no centering — calibration reference handles alignment)
    kp_mm = kp_px * mm_per_pixel  # (11, 2)

    if phase == "calibration":
        # --- Calibration: accumulate frames, emit rest-pose mesh ---
        if "cal_frames" not in state:
            state["cal_frames"] = []
            state["cal_jaw_openings"] = []

        state["cal_frames"].append(kp_mm.copy())
        jaw = state.get("mouth_open_norm")
        if jaw is not None:
            state["cal_jaw_openings"].append(jaw)

        pose_pts = state.get("face_pose_pts")
        if pose_pts is not None:
            if "cal_pose_pts" not in state:
                state["cal_pose_pts"] = []
                state["cal_mouth_open"] = []
            state["cal_pose_pts"].append(pose_pts.copy())
            state["cal_mouth_open"].append(state.get("mouth_open_norm", 0.0))

        n = len(state["cal_frames"])
        min_frames = config.get("calibration_min_frames", 30)
        if n == 1 or n % 10 == 0:
            log.info("calibration: %d/%d frames", n, min_frames)

        _emit_rest_mesh(state, config, _display_callback)
        if _display_callback:
            _display_callback(display_id, "mandible_angle", state.get("jaw_close_offset", 0.0))
        return

    # --- Tracking: displacement from calibration reference ---
    ref_kp_mm = state.get("ref_kp_mm")
    if ref_kp_mm is None:
        # No calibration data — fall back to rest pose
        _emit_rest_mesh(state, config, _display_callback)
        return

    # Displacement in US mm-space
    delta_mm = kp_mm - ref_kp_mm  # (11, 2)

    # Map displacement to model space, add to rest positions — these are the
    # *raw DLC-derived* targets, before any anatomical constraints.
    rest_pos = state["dorsal_rest_positions"]
    dlc_targets = rest_pos.copy()
    dlc_targets[:, 2] += delta_mm[:, 0]     # US X disp → model Z
    dlc_targets[:, 1] += -delta_mm[:, 1]    # US Y disp → model -Y

    # Spatial spline smoothing (treat keypoints as spline handles)
    spline_s = config.get("spline_smoothing", 0.5)
    if spline_s > 0:
        dlc_targets = _smooth_targets_spline(dlc_targets, smoothing=spline_s)

    # Anatomical-target pipeline: confidence-weighted blend, per-bone
    # stiffness ramp (root stiff, tip free), arc-length conservation.
    # See sigflow/nodes/tongue_targets.py for the full rationale.
    anatomical_params = AnatomicalTargetParams(
        confidence_threshold=config.get("confidence_threshold", 0.1),
        confidence_soft_range=config.get("confidence_soft_range", 0.4),
        stiffness_root=config.get("stiffness_root", 0.5),
        stiffness_tip=config.get("stiffness_tip", 0.1),
        max_displacement_root_mm=config.get("max_displacement_root_mm", 8.0),
        max_displacement_tip_mm=config.get("max_displacement_tip_mm", 25.0),
        arc_length_min_ratio=config.get("arc_length_min_ratio", 0.92),
        arc_length_max_ratio=config.get("arc_length_max_ratio", 1.08),
    )
    target = compute_anatomical_targets(
        dlc_targets=dlc_targets,
        rest_positions=rest_pos,
        confidences=confidence,
        rest_bone_lengths=state["rest_bone_lengths"],
        params=anatomical_params,
    )

    # Head rotation (SVD Procrustes delta from calibration)
    ref_pose = state.get("ref_pose_pts")
    cur_pose = state.get("face_pose_pts")
    if ref_pose is not None and cur_pose is not None:
        head_euler = _compute_head_rotation(ref_pose, cur_pose)
    else:
        head_euler = np.zeros(3, dtype=np.float32)

    # Mandible angle from mouth opening delta (offset from closed-jaw position)
    mouth_delta = state.get("mouth_open_norm", 0.0) - state.get("ref_mouth_open", 0.0)
    raw_opening = max(0.0, mouth_delta) * config.get("mandible_angle_scale", 100.0)
    jaw_close = state.get("jaw_close_offset", 0.0)
    mandible_angle = float(jaw_close + raw_opening)

    # Coupled protrusion: condyle slides forward as mouth opens
    coupled_factor = config.get("tmj_coupled_factor", 0.3)
    protrusion = float(coupled_factor * raw_opening)

    # Temporal smoothing (one-euro filter)
    min_cutoff = config.get("smooth_min_cutoff", 1.0)
    beta = config.get("smooth_beta", 0.007)
    if t > 0:
        head_euler = _one_euro_filter("head_rot", state, head_euler, t, min_cutoff, beta, d_cutoff=1.0)
        mandible_angle = float(_one_euro_filter(
            "mandible", state, np.array([mandible_angle]), t, min_cutoff, beta, d_cutoff=1.0
        )[0])
        protrusion = float(_one_euro_filter(
            "protrusion", state, np.array([protrusion]), t, min_cutoff, beta, d_cutoff=1.0
        )[0])
        flat = target.ravel()
        flat = _one_euro_filter("smooth", state, flat, t, min_cutoff, beta, d_cutoff=1.0)
        target = flat.reshape(11, 3)

    # Curvature + min-bone-distance constraints now run inside
    # compute_anatomical_targets above (single source of truth for both
    # CPU LBS and GPU Skin paths).

    # Compute bone world transforms (connected-chain FK with preserved bone lengths)
    bone_transforms = _compute_bone_transforms(
        target, state["bone_rest_world"], state["rest_bone_lengths"]
    )

    # Apply LBS skinning
    deformed_verts, deformed_normals = _apply_lbs(
        state["vertices"], state["normals"],
        state["joint_indices"], state["joint_weights"],
        bone_transforms, state["inv_bind_matrices"],
    )

    # Transform entire tongue into jaw-space (DLC offsets are probe-relative ≈ jaw-relative)
    if mandible_angle != 0.0 or protrusion != 0.0:
        tmj = state["tmj_position"]
        a = np.radians(mandible_angle)
        cos_a, sin_a = np.cos(a), np.sin(a)
        Rx = np.array([[1, 0, 0], [0, cos_a, -sin_a], [0, sin_a, cos_a]], dtype=np.float32)

        deformed_verts = (deformed_verts - tmj) @ Rx.T
        deformed_verts[:, 2] += protrusion
        deformed_verts += tmj

        deformed_normals = deformed_normals @ Rx.T

    # Hide ventral surface
    if "vertex_dorsalness" in state:
        deformed_verts = _collapse_ventral(deformed_verts, state["vertex_dorsalness"])

    # Pack interleaved vertex buffer: [pos3 + normal3 [+ uv2]] × V, all float32
    uvs = state.get("uvs")
    if uvs is not None:
        interleaved = np.empty((state["num_vertices"], 8), dtype=np.float32)
        interleaved[:, :3] = deformed_verts
        interleaved[:, 3:6] = deformed_normals
        interleaved[:, 6:] = uvs
    else:
        interleaved = np.empty((state["num_vertices"], 6), dtype=np.float32)
        interleaved[:, :3] = deformed_verts
        interleaved[:, 3:] = deformed_normals

    if _display_callback:
        _display_callback(display_id, "mesh", interleaved.tobytes())
        _display_callback(display_id, "head_rotation", head_euler)
        _display_callback(display_id, "mandible_angle", mandible_angle)
        _display_callback(display_id, "mandible_protrusion", protrusion)
