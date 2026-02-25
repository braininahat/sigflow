"""3D tongue model sink — drives a skinned GLB mesh from DLC keypoints.

Receives keypoint predictions, converts pixel→mm space (absolute calibration),
applies one-euro temporal smoothing and anatomical constraints, computes bone
world transforms, applies linear blend skinning in numpy, and pushes deformed
vertex buffers to the display callback for QML rendering.

Dorsal chain (joints 0–10) maps 1:1 to DLC keypoints 0–10.
Ventral chain (joints 11–18) stays at rest pose.
"""
import logging
from pathlib import Path

import numpy as np

from sigflow.node import sink_node, Param
from sigflow.types import Port, Keypoints

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


def _keypoints_to_mm_space(keypoints_px, mm_per_pixel, rest_bone_positions):
    """Convert DLC pixel keypoints to mm-space positions.

    Fixed transform — no per-frame scale fitting. Uses absolute mm calibration
    from the ultrasound probe's mm_per_pixel.

    Args:
        keypoints_px: (11, 2) pixel coordinates for dorsal keypoints
        mm_per_pixel: scalar conversion factor from probe calibration
        rest_bone_positions: (11, 3) mm-space rest-pose positions of dorsal bones

    Returns:
        (11, 3) mm-space target positions
    """
    # Pixel → mm (absolute calibration)
    kp_mm = keypoints_px * mm_per_pixel

    # Axis mapping: US image X → model Z, US image Y → model -Y
    # Center mm-space keypoints on the mm-space rest-pose bone chain center
    rest_z_center = (rest_bone_positions[0, 2] + rest_bone_positions[-1, 2]) / 2.0
    rest_y_center = (rest_bone_positions[:, 1].min() + rest_bone_positions[:, 1].max()) / 2.0

    kp_x_center = (kp_mm[:, 0].min() + kp_mm[:, 0].max()) / 2.0
    kp_y_center = (kp_mm[:, 1].min() + kp_mm[:, 1].max()) / 2.0

    target = np.zeros((11, 3), dtype=np.float32)
    target[:, 2] = (kp_mm[:, 0] - kp_x_center) + rest_z_center   # US X → model Z
    target[:, 1] = -(kp_mm[:, 1] - kp_y_center) + rest_y_center  # US Y → model -Y
    target[:, 0] = rest_bone_positions[:, 0].mean()  # X stays at midsagittal plane

    return target


def _apply_constraints(target_positions, max_angle_deg=60.0, min_bone_dist=1.0):
    """Apply anatomical constraints to target bone positions (in-place).

    - Max curvature: limit angle between consecutive bone segments
    - Min bone distance: prevent adjacent targets from collapsing
    """
    n = len(target_positions)
    cos_max = np.cos(np.radians(max_angle_deg))

    # Min bone distance — push apart if too close
    for i in range(n - 1):
        d = target_positions[i + 1] - target_positions[i]
        dist = np.linalg.norm(d)
        if dist < min_bone_dist:
            if dist < 1e-6:
                # Degenerate — nudge along Z
                d = np.array([0.0, 0.0, min_bone_dist], dtype=np.float32)
            else:
                d = d / dist * min_bone_dist
            target_positions[i + 1] = target_positions[i] + d

    # Max curvature — clamp angle between consecutive segments
    for i in range(1, n - 1):
        v1 = target_positions[i] - target_positions[i - 1]
        v2 = target_positions[i + 1] - target_positions[i]
        n1 = np.linalg.norm(v1)
        n2 = np.linalg.norm(v2)
        if n1 < 1e-6 or n2 < 1e-6:
            continue
        v1_hat = v1 / n1
        v2_hat = v2 / n2
        cos_angle = np.dot(v1_hat, v2_hat)
        if cos_angle < cos_max:
            # Rotate v2 toward v1 direction until angle = max_angle
            # Project v2 onto plane perpendicular to v1, blend
            axis = np.cross(v1_hat, v2_hat)
            axis_norm = np.linalg.norm(axis)
            if axis_norm < 1e-8:
                continue
            axis /= axis_norm
            sin_max = np.sin(np.radians(max_angle_deg))
            new_dir = v1_hat * cos_max + np.cross(axis, v1_hat) * sin_max
            new_dir /= np.linalg.norm(new_dir) + 1e-12
            target_positions[i + 1] = target_positions[i] + new_dir * n2

    return target_positions


def _compute_bone_transforms(target_positions, rest_transforms, num_dorsal=11):
    """Compute world transforms for all joints from dorsal target positions.

    Args:
        target_positions: (11, 3) mm-space positions for dorsal bones
        rest_transforms: (J, 4, 4) rest-pose world transforms
        num_dorsal: number of dorsal chain bones (default 11)

    Returns:
        (J, 4, 4) new world transforms
    """
    num_joints = rest_transforms.shape[0]
    transforms = rest_transforms.copy()

    prev_direction = np.array([0.0, 1.0, 0.0], dtype=np.float32)

    for i in range(num_dorsal):
        pos = target_positions[i]
        if i < num_dorsal - 1:
            direction = target_positions[i + 1] - target_positions[i]
            norm = np.linalg.norm(direction)
            if norm > 1e-6:
                direction /= norm
                prev_direction = direction
            else:
                direction = prev_direction
        else:
            direction = prev_direction

        transforms[i] = _build_look_along_y(pos, direction)

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

    # Recompute inverse bind matrices from scaled rest world
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

    # Cache dorsal rest positions (now in mm)
    state["dorsal_rest_positions"] = bone_rest_world[:11, :3, 3].copy()

    # Precompute index buffer bytes (static, sent once)
    state["index_bytes"] = mesh["indices"].astype(np.uint32).tobytes()
    state["indices_sent"] = False

    # Send static jaw meshes (scaled to mm) once
    display_id = config["display_id"]
    static_meshes = mesh.get("static_meshes", {})
    if static_meshes and _display_callback:
        jaw_data = {}
        for name, smesh in static_meshes.items():
            scaled_verts = smesh["vertices"] * mm_per_model_unit
            scaled_normals = smesh["normals"]
            # Pack interleaved: [pos3 + normal3] × V
            n_verts = len(scaled_verts)
            interleaved = np.empty((n_verts, 6), dtype=np.float32)
            interleaved[:, :3] = scaled_verts
            interleaved[:, 3:] = scaled_normals
            jaw_data[name] = (interleaved.tobytes(), smesh["indices"].astype(np.uint32).tobytes())
            log.info("static mesh '%s': %d vertices", name, n_verts)
        _display_callback(display_id, "mesh_static", jaw_data)

    log.info("tongue model loaded: %d vertices, %d joints (mm-space)",
             mesh["num_vertices"], mesh["num_joints"])


@sink_node(
    name="tongue_model_display",
    inputs=[Port("keypoints", Keypoints)],
    category="visualization",
    params=[
        Param("display_id", "str", "tongue_model", label="Display Target"),
        Param("model_path", "str", "assets/TongueBond.glb", label="Model Path"),
        Param("confidence_threshold", "float", 0.1, label="Min Confidence"),
        Param("tongue_length_mm", "float", 70.0, label="Reference Tongue Length (mm)"),
        Param("smooth_min_cutoff", "float", 1.0, label="Smoothing (lower=smoother)"),
        Param("smooth_beta", "float", 0.007, label="Smoothing Speed Adapt"),
    ],
)
def tongue_model_display(item, *, state, config):
    from sigflow.nodes.app_display import _display_callback

    if "vertices" not in state:
        _init_mesh(state, config)

    display_id = config["display_id"]

    # Send static index buffer once
    if not state["indices_sent"] and _display_callback:
        _display_callback(display_id, "mesh_indices", state["index_bytes"])
        state["indices_sent"] = True

    keypoints = item.data  # (16, 3) → [x, y, confidence]
    if keypoints is None or len(keypoints) < 11:
        return

    # Extract dorsal keypoints (first 11) and filter by confidence
    dorsal_kp = keypoints[:11]
    confidence = dorsal_kp[:, 2]
    threshold = config["confidence_threshold"]

    valid = confidence >= threshold
    if valid.sum() < 3:
        return  # Not enough keypoints to drive the model

    # Get mm_per_pixel from metadata
    mm_per_pixel = item.metadata.get("mm_per_pixel", 1.0)
    if isinstance(mm_per_pixel, (list, tuple)):
        mm_per_pixel = mm_per_pixel[0]  # Use X if separate X/Y

    # For invalid keypoints, interpolate from neighbors
    kp_px = dorsal_kp[:, :2].copy()
    if not valid.all():
        valid_indices = np.where(valid)[0]
        invalid_indices = np.where(~valid)[0]
        for idx in invalid_indices:
            left = valid_indices[valid_indices < idx]
            right = valid_indices[valid_indices > idx]
            if len(left) > 0 and len(right) > 0:
                l, r = left[-1], right[0]
                t = (idx - l) / (r - l)
                kp_px[idx] = kp_px[l] * (1 - t) + kp_px[r] * t
            elif len(left) > 0:
                kp_px[idx] = kp_px[left[-1]]
            elif len(right) > 0:
                kp_px[idx] = kp_px[right[0]]

    # Pixel → mm space (absolute calibration, no per-frame fitting)
    target_positions = _keypoints_to_mm_space(
        kp_px, mm_per_pixel, state["dorsal_rest_positions"]
    )

    # Temporal smoothing (one-euro filter)
    min_cutoff = config.get("smooth_min_cutoff", 1.0)
    beta = config.get("smooth_beta", 0.007)
    t = item.lsl_timestamp if hasattr(item, "lsl_timestamp") else 0.0
    if t > 0:
        flat = target_positions.ravel()
        flat = _one_euro_filter("smooth", state, flat, t, min_cutoff, beta, d_cutoff=1.0)
        target_positions = flat.reshape(11, 3)

    # Anatomical constraints
    target_positions = _apply_constraints(target_positions)

    # Compute bone world transforms
    bone_transforms = _compute_bone_transforms(
        target_positions, state["bone_rest_world"]
    )

    # Apply LBS skinning
    deformed_verts, deformed_normals = _apply_lbs(
        state["vertices"],
        state["normals"],
        state["joint_indices"],
        state["joint_weights"],
        bone_transforms,
        state["inv_bind_matrices"],
    )

    # Pack interleaved vertex buffer: [pos3 + normal3] × V, all float32
    interleaved = np.empty((state["num_vertices"], 6), dtype=np.float32)
    interleaved[:, :3] = deformed_verts
    interleaved[:, 3:] = deformed_normals
    vertex_bytes = interleaved.tobytes()

    if _display_callback:
        _display_callback(display_id, "mesh", vertex_bytes)
