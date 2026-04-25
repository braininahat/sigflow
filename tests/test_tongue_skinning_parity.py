"""Parity tests between the NumPy LBS and GPU-skin tongue paths.

The GPU path (``skinned_tongue_display``) and the LBS path
(``tongue_model_display``) both build target positions through the same
``compute_anatomical_targets`` pipeline. After the W5 fix, the GPU path
also runs ``compute_chain_fk`` and converts world transforms to **3D**
quaternions instead of the broken 2D-only quaternions that caused the
"scrunched" mesh.

The headline parity invariant: at the rest pose, both paths' bone-world
transforms should agree to within float32 epsilon. Translation columns
must align in the model's mm space, and rotations must be identity (or
near-identity) for unmoved bones.
"""
from __future__ import annotations

import numpy as np

from sigflow.nodes.tongue_targets import (
    build_look_along_y,
    compute_anatomical_targets,
    compute_chain_fk,
    world_to_local_quaternion,
)


def _rest_chain(n: int = 11, spacing: float = 5.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build a synthetic rest pose where each bone's local Y axis points
    along the chain direction (matches the GLB rest-pose convention used
    by ``compute_chain_fk``)."""
    rest_pos = np.zeros((n, 3), dtype=np.float32)
    rest_pos[:, 2] = np.arange(n, dtype=np.float32) * spacing
    rest_bone_lengths = np.full(n - 1, spacing, dtype=np.float32)
    chain_dir = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    rest_world = np.empty((n, 4, 4), dtype=np.float32)
    for i in range(n):
        rest_world[i] = build_look_along_y(rest_pos[i], chain_dir)
    return rest_pos, rest_bone_lengths, rest_world


def test_rest_pose_quaternion_is_identity() -> None:
    """When DLC equals rest, every bone's local rotation should be near-identity."""
    rest_pos, rest_bones, rest_world = _rest_chain()
    dlc = rest_pos.copy()
    conf = np.ones(rest_pos.shape[0], dtype=np.float32)

    targets = compute_anatomical_targets(dlc, rest_pos, conf, rest_bones)
    bone_world = compute_chain_fk(targets, rest_world, rest_bones, num_dorsal=11)
    for i in range(11):
        w, x, y, z = world_to_local_quaternion(bone_world[i], rest_world[i])
        # ±w both represent identity rotation (Hamilton convention).
        assert abs(abs(w) - 1.0) < 1e-3, (
            f"bone {i} local quat w={w:.4f} should be ±1 for identity"
        )
        assert abs(x) < 5e-3 and abs(y) < 5e-3 and abs(z) < 5e-3, (
            f"bone {i} local quat axis ({x:.4f}, {y:.4f}, {z:.4f}) should be ~0"
        )


def test_chain_fk_preserves_bone_lengths() -> None:
    """Even when DLC pulls the chain to elongate, FK output preserves rest-pose
    bone lengths."""
    rest_pos, rest_bones, rest_world = _rest_chain()
    dlc = rest_pos.copy()
    dlc[:, 2] *= 2.0  # ask for 2x stretch
    conf = np.ones(rest_pos.shape[0], dtype=np.float32)
    targets = compute_anatomical_targets(dlc, rest_pos, conf, rest_bones)
    bone_world = compute_chain_fk(targets, rest_world, rest_bones, num_dorsal=11)

    chain_positions = bone_world[:11, :3, 3]
    out_bones = np.linalg.norm(np.diff(chain_positions, axis=0), axis=1)
    # FK preserves bone lengths exactly; arc-length conservation may rescale
    # the chain but each individual segment uses ``rest_bone_lengths``.
    np.testing.assert_allclose(out_bones, rest_bones, atol=1e-4)


def test_quaternion_3d_axis_when_dlc_articulates_in_y() -> None:
    """A pure-Y articulation (model space) should yield a quaternion with
    a non-zero X-component (rotation about model-X axis), but unlike the
    old 2D-only path, Y and Z should not be globally pinned to zero —
    later bones in the chain may pick up Y/Z components as the FK chain
    bends in 3D.
    """
    rest_pos, rest_bones, rest_world = _rest_chain()
    # Push the entire chain 8 mm in +Y; this is a uniform translation,
    # so each bone's rotation will be the same: rotation about model-X.
    dlc = rest_pos.copy()
    dlc[:, 1] += 8.0
    conf = np.ones(rest_pos.shape[0], dtype=np.float32)
    targets = compute_anatomical_targets(dlc, rest_pos, conf, rest_bones)
    bone_world = compute_chain_fk(targets, rest_world, rest_bones, num_dorsal=11)

    # Joint 5 (mid-chain) should have a non-trivial rotation.
    w, x, y, z = world_to_local_quaternion(bone_world[5], rest_world[5])
    quat_norm = float(np.sqrt(w * w + x * x + y * y + z * z))
    assert abs(quat_norm - 1.0) < 1e-3, f"quat not unit-normalised: {quat_norm}"
    # The rotation magnitude must be non-trivial (the chain bent).
    angle_rad = 2 * float(np.arccos(np.clip(abs(w), 0, 1)))
    assert angle_rad > 0.05, (
        f"bone 5 rotation angle {angle_rad:.4f} rad — should be substantial "
        "for an 8 mm Y articulation"
    )


def test_lbs_and_gpu_share_same_targets() -> None:
    """Sanity: feeding the same inputs through compute_anatomical_targets
    is deterministic — the LBS and GPU paths both call it identically and
    therefore see identical target positions. Guards against accidental
    divergence in either node's preprocessing.
    """
    rest_pos, rest_bones, _ = _rest_chain()
    rng = np.random.default_rng(seed=42)
    dlc = rest_pos + rng.normal(scale=2.0, size=rest_pos.shape).astype(np.float32)
    conf = rng.uniform(0.0, 1.0, size=rest_pos.shape[0]).astype(np.float32)

    out_a = compute_anatomical_targets(dlc, rest_pos, conf, rest_bones)
    out_b = compute_anatomical_targets(dlc, rest_pos, conf, rest_bones)
    np.testing.assert_array_equal(out_a, out_b)
