"""Shared anatomical-target pipeline for tongue mesh deformation.

Both the NumPy-LBS path (``tongue_model_display``) and the GPU-skinned path
(``skinned_tongue_display``) need to derive a smooth, anatomically-plausible
sequence of dorsal target positions from noisy DLC keypoints. This module
is the single source of truth for that derivation so the two skinning
paths can never disagree.

The pipeline runs in this order each frame:

1. **Confidence-weighted target blend** — replaces the historical binary
   gate. Per-joint blend factor ``w_i = clip((conf_i - thresh) / soft, 0, 1)``.
   The DLC target is mixed with a "rest + neighbour-drift" anchor, so a
   joint with conf=0.05 follows what its neighbours are doing instead of
   snapping back to rest. This is the dominant fix for "floppy" articulation
   under noisy DLC.

2. **Per-bone stiffness & displacement ramp** — root bones stiff
   (stiffness ≈ 0.5, max displacement ≈ 8 mm), tip bones free
   (stiffness ≈ 0.1, max displacement ≈ 25 mm). The previous global
   scalar treated every bone equally, which is anatomically wrong: the
   tongue body is anchored, the tip is mobile.

3. **Arc-length / volume conservation** — clamps the total chain arc length
   to ``[min_ratio · L_rest, max_ratio · L_rest]`` (default ±8 %). This is
   the cheapest hydrostatic analogue: the tongue is roughly incompressible
   so its midline arc length is roughly constant. Drawn from MyoSim3D's
   muscular-hydrostat model (``sigflow.biomech.forward_solver``) without
   reaching into that runtime path.

4. **Curvature limit + min bone distance** — clamps the angle between
   consecutive segments to ``max_angle_deg`` and pushes adjacent targets
   apart if they're closer than ``min_bone_dist``. Catches edge cases the
   per-bone rigidity ramp can't (two high-confidence neighbours that
   disagree on direction, or near-coincident keypoints).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class AnatomicalTargetParams:
    """All knobs exposed via Param(...) at the node level."""
    # Confidence-weighted blend
    confidence_threshold: float = 0.1
    confidence_soft_range: float = 0.4
    # Per-bone stiffness ramp: linspace(stiffness_root, stiffness_tip, num_dorsal)
    stiffness_root: float = 0.5
    stiffness_tip: float = 0.1
    # Per-bone max displacement ramp (mm)
    max_displacement_root_mm: float = 8.0
    max_displacement_tip_mm: float = 25.0
    # Arc-length conservation ratios
    arc_length_min_ratio: float = 0.92
    arc_length_max_ratio: float = 1.08


def confidence_weighted_blend(
    dlc_targets: np.ndarray,
    rest_positions: np.ndarray,
    confidences: np.ndarray,
    threshold: float = 0.1,
    soft_range: float = 0.4,
) -> np.ndarray:
    """Blend DLC-predicted targets with rest+neighbour-drift anchors per joint.

    Args:
        dlc_targets: (n, 3) target positions derived from DLC keypoints
        rest_positions: (n, 3) rest-pose joint positions (model space)
        confidences: (n,) DLC confidences in [0, 1]
        threshold: confidence below this is treated as "no signal"
        soft_range: confidence range above ``threshold`` over which the
            DLC contribution ramps from 0 to 1

    Returns:
        (n, 3) blended targets — confident joints follow DLC, low-confidence
        joints follow their neighbours' drift from rest, fully unreliable
        joints fall back to rest pose.

    Why this beats binary interpolation:
        Linear interpolation between high-confidence neighbours throws
        away whatever (weak) signal the low-confidence joint has, and
        produces a kink whenever a neighbour is also weak. The drift
        anchor — the average displacement of the two adjacent joints
        weighted by *their* confidences — preserves smooth chain motion
        even when half the keypoints are unreliable.
    """
    n = dlc_targets.shape[0]
    if soft_range <= 0:
        soft_range = 1e-3

    w = np.clip((confidences - threshold) / soft_range, 0.0, 1.0).astype(np.float32)

    # Neighbour drift: per joint i, average of valid-weighted displacements
    # at i-1 and i+1.  Joints at the chain ends use only the available side.
    rest_drift = dlc_targets - rest_positions  # (n, 3)
    weighted_drift = rest_drift * w[:, None]
    sum_weights = np.zeros(n, dtype=np.float32)
    sum_drift = np.zeros((n, 3), dtype=np.float32)
    for i in range(n):
        if i > 0:
            sum_drift[i] += weighted_drift[i - 1]
            sum_weights[i] += w[i - 1]
        if i < n - 1:
            sum_drift[i] += weighted_drift[i + 1]
            sum_weights[i] += w[i + 1]
    neighbour_drift = np.where(
        sum_weights[:, None] > 1e-6,
        sum_drift / np.maximum(sum_weights[:, None], 1e-6),
        np.zeros_like(sum_drift),
    )

    # Composite anchor: rest + neighbour drift (smooth chain follower)
    anchor = rest_positions + neighbour_drift

    # Final blend
    return (w[:, None] * dlc_targets + (1.0 - w[:, None]) * anchor).astype(np.float32)


def per_bone_rigidity(
    targets: np.ndarray,
    rest_positions: np.ndarray,
    stiffness: np.ndarray,
    max_displacement_mm: np.ndarray,
) -> np.ndarray:
    """Per-bone rigidity + max-displacement clamp.

    Replaces the historical scalar ``_apply_rigidity`` for the dorsal
    chain. Pass per-bone arrays so root bones can be stiffer than the tip.

    Args:
        targets: (n, 3) candidate target positions
        rest_positions: (n, 3) rest-pose positions
        stiffness: (n,) per-bone stiffness in [0, 1]; 0 = follow target,
            1 = lock to rest
        max_displacement_mm: (n,) per-bone displacement clamp in mm

    Returns:
        (n, 3) constrained targets
    """
    stiffness = stiffness.astype(np.float32)
    max_displacement_mm = max_displacement_mm.astype(np.float32)

    # Rest-pose blend: (1 - s) * target + s * rest
    blended = (1.0 - stiffness)[:, None] * targets + stiffness[:, None] * rest_positions

    # Max displacement clamp per joint
    delta = blended - rest_positions
    dist = np.linalg.norm(delta, axis=1)
    over = dist > max_displacement_mm
    if over.any():
        # Scale down only the offending joints; leave clean ones untouched.
        scale = np.ones_like(dist)
        valid = (dist > 1e-6) & over
        scale[valid] = max_displacement_mm[valid] / dist[valid]
        blended = rest_positions + delta * scale[:, None]
    return blended.astype(np.float32)


def arc_length_conserve(
    targets: np.ndarray,
    rest_bone_lengths: np.ndarray,
    min_ratio: float = 0.92,
    max_ratio: float = 1.08,
) -> np.ndarray:
    """Clamp total chain arc length to a fraction of the rest-pose arc length.

    The tongue is roughly incompressible: its midline arc length is roughly
    constant during articulation (the muscular-hydrostat property). When
    DLC noise pushes the chain to elongate or compress beyond a tolerance,
    rescale every segment uniformly so the total arc length lands inside
    ``[min_ratio · L_rest, max_ratio · L_rest]``.

    Note this preserves segment ratios, not segment lengths. The downstream
    chain-FK step then reinterprets the rescaled targets in terms of
    rest-pose bone lengths, which keeps individual bone lengths rigid while
    letting the chain *direction* track the rescaled targets — exactly the
    rigid-anchor behaviour we want.

    Args:
        targets: (n, 3) candidate target positions
        rest_bone_lengths: (n-1,) rest-pose bone lengths
        min_ratio: lower bound of allowed arc length ratio
        max_ratio: upper bound

    Returns:
        (n, 3) targets with total arc length inside the allowed range
    """
    n = targets.shape[0]
    if n < 2:
        return targets
    seg = np.diff(targets, axis=0)
    seg_len = np.linalg.norm(seg, axis=1)
    L_target = float(np.sum(seg_len))
    L_rest = float(np.sum(rest_bone_lengths))
    if L_rest <= 0 or L_target <= 0:
        return targets

    ratio = L_target / L_rest
    if min_ratio <= ratio <= max_ratio:
        return targets

    clamped_ratio = float(np.clip(ratio, min_ratio, max_ratio))
    scale = clamped_ratio / ratio  # < 1 if too long, > 1 if too short
    # Walk the chain from joint 0, scaling each segment.
    out = np.empty_like(targets)
    out[0] = targets[0]
    for i in range(1, n):
        d = targets[i] - targets[i - 1]
        out[i] = out[i - 1] + d * scale
    return out.astype(np.float32)


def apply_chain_constraints(
    target_positions: np.ndarray,
    max_angle_deg: float = 60.0,
    min_bone_dist: float = 1.0,
) -> np.ndarray:
    """Clamp inter-segment angle and push adjacent targets apart.

    Two passes, in order:

    1. **Min bone distance** — if consecutive targets are closer than
       ``min_bone_dist``, push the second one outward along the segment
       direction (or along world Z if degenerate).
    2. **Max curvature** — if the angle between two consecutive segments
       exceeds ``max_angle_deg``, rotate the second segment toward the
       first until the angle equals the cap.

    Operates in-place on the input array (also returned for chaining).
    """
    n = len(target_positions)
    cos_max = np.cos(np.radians(max_angle_deg))

    for i in range(n - 1):
        d = target_positions[i + 1] - target_positions[i]
        dist = np.linalg.norm(d)
        if dist < min_bone_dist:
            if dist < 1e-6:
                d = np.array([0.0, 0.0, min_bone_dist], dtype=np.float32)
            else:
                d = d / dist * min_bone_dist
            target_positions[i + 1] = target_positions[i] + d

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


def build_look_along_y(position: np.ndarray, direction: np.ndarray) -> np.ndarray:
    """4×4 transform with Y axis along ``direction``, origin at ``position``.

    Same convention as the GLB rest pose (forward axis = +Y per joint).
    Identical math to ``tongue_model_display._build_look_along_y`` —
    factored here so both skinning paths share one source of truth.
    """
    y = direction / (np.linalg.norm(direction) + 1e-12)
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


def compute_chain_fk(
    target_positions: np.ndarray,
    rest_transforms: np.ndarray,
    rest_bone_lengths: np.ndarray,
    num_dorsal: int = 11,
) -> np.ndarray:
    """Connected-chain FK with preserved bone lengths.

    Each joint's position is determined by the previous joint's position
    plus the direction toward this joint's target × rest-pose bone length.
    Bone lengths stay rigid (no stretch/compress); chain *direction*
    follows the targets. Identical to ``tongue_model_display._compute_bone_transforms``.

    Args:
        target_positions: (n, 3) target positions for dorsal joints
        rest_transforms: (J, 4, 4) rest-pose world transforms (≥ ``num_dorsal``)
        rest_bone_lengths: (n-1,) rest-pose distances between consecutive joints
        num_dorsal: number of dorsal-chain bones

    Returns:
        (J, 4, 4) world transforms — dorsal chain articulated, rest stays at rest
    """
    transforms = rest_transforms.copy()

    chain = np.empty((num_dorsal, 3), dtype=np.float32)
    chain[0] = target_positions[0]

    prev_direction = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    for i in range(1, num_dorsal):
        direction = target_positions[i] - chain[i - 1]
        norm = np.linalg.norm(direction)
        if norm > 1e-6:
            direction /= norm
            prev_direction = direction
        else:
            direction = prev_direction
        chain[i] = chain[i - 1] + direction * rest_bone_lengths[i - 1]

    for i in range(num_dorsal):
        if i < num_dorsal - 1:
            direction = chain[i + 1] - chain[i]
            norm = np.linalg.norm(direction)
            if norm > 1e-6:
                direction /= norm
            else:
                direction = prev_direction
        else:
            direction = chain[i] - chain[i - 1]
            norm = np.linalg.norm(direction)
            if norm > 1e-6:
                direction /= norm
            else:
                direction = prev_direction
        transforms[i] = build_look_along_y(chain[i], direction)
    return transforms


def world_to_local_quaternion(
    world: np.ndarray,
    rest_world: np.ndarray,
) -> tuple[float, float, float, float]:
    """Local rotation quaternion of ``world`` relative to ``rest_world``.

    Computes ``R_rest⁻¹ · R_world`` (rest is orthonormal so transpose =
    inverse) and returns it as a (w, x, y, z) Hamilton quaternion. Both
    inputs are 4×4 row-major; only the upper-left 3×3 is used. Kept as
    a thin wrapper around :func:`rotation_matrix_to_quaternion` for
    callers that still want the world→local form.
    """
    R_local = rest_world[:3, :3].T @ world[:3, :3]
    return rotation_matrix_to_quaternion(R_local)


def rotation_matrix_to_quaternion(R: np.ndarray) -> tuple[float, float, float, float]:
    """Convert a 3×3 rotation matrix to a (w, x, y, z) Hamilton quaternion.

    Shepperd's method (numerically stable across all rotation regimes).
    The returned quaternion is normalized.
    """
    m00, m01, m02 = R[0]
    m10, m11, m12 = R[1]
    m20, m21, m22 = R[2]
    trace = m00 + m11 + m22
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (m21 - m12) * s
        y = (m02 - m20) * s
        z = (m10 - m01) * s
    elif m00 > m11 and m00 > m22:
        s = 2.0 * np.sqrt(1.0 + m00 - m11 - m22)
        w = (m21 - m12) / s
        x = 0.25 * s
        y = (m01 + m10) / s
        z = (m02 + m20) / s
    elif m11 > m22:
        s = 2.0 * np.sqrt(1.0 + m11 - m00 - m22)
        w = (m02 - m20) / s
        x = (m01 + m10) / s
        y = 0.25 * s
        z = (m12 + m21) / s
    else:
        s = 2.0 * np.sqrt(1.0 + m22 - m00 - m11)
        w = (m10 - m01) / s
        x = (m02 + m20) / s
        y = (m12 + m21) / s
        z = 0.25 * s
    n = float(np.sqrt(w * w + x * x + y * y + z * z) + 1e-12)
    return float(w / n), float(x / n), float(y / n), float(z / n)


def compute_anatomical_targets(
    dlc_targets: np.ndarray,
    rest_positions: np.ndarray,
    confidences: np.ndarray,
    rest_bone_lengths: np.ndarray,
    params: Optional[AnatomicalTargetParams] = None,
) -> np.ndarray:
    """Run the full anatomical-target pipeline in canonical order.

    Order matters: confidence blend first (so unreliable joints don't
    trigger displacement clamps spuriously), then per-bone rigidity, then
    arc-length conservation. The downstream caller (LBS or GPU skin) then
    layers existing curvature / min-distance constraints + FK on top.
    """
    if params is None:
        params = AnatomicalTargetParams()
    n = dlc_targets.shape[0]

    targets = confidence_weighted_blend(
        dlc_targets=dlc_targets,
        rest_positions=rest_positions,
        confidences=confidences,
        threshold=params.confidence_threshold,
        soft_range=params.confidence_soft_range,
    )

    stiffness = np.linspace(
        params.stiffness_root, params.stiffness_tip, n, dtype=np.float32,
    )
    max_disp = np.linspace(
        params.max_displacement_root_mm,
        params.max_displacement_tip_mm,
        n,
        dtype=np.float32,
    )
    targets = per_bone_rigidity(targets, rest_positions, stiffness, max_disp)

    targets = arc_length_conserve(
        targets,
        rest_bone_lengths,
        min_ratio=params.arc_length_min_ratio,
        max_ratio=params.arc_length_max_ratio,
    )
    targets = apply_chain_constraints(targets)
    return targets
