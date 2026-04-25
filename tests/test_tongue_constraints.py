"""Tests for sigflow.nodes.tongue_targets — anatomical-target pipeline.

Three properties we care about:
1. Rest-pose fidelity: when DLC predicts the rest pose with full
   confidence, the output equals rest within float epsilon.
2. Arc-length conservation: when DLC predicts a deformation that elongates
   or compresses the chain beyond ±max_ratio, the output is rescaled to
   land inside that range.
3. Low-confidence neighbour-following: when a single keypoint has very
   low confidence, the output for that joint follows its (high-confidence)
   neighbours' drift instead of snapping back to rest.
"""
from __future__ import annotations

import numpy as np

from sigflow.nodes.tongue_targets import (
    AnatomicalTargetParams,
    arc_length_conserve,
    compute_anatomical_targets,
    confidence_weighted_blend,
    per_bone_rigidity,
)


def _rest_chain(n: int = 11, spacing: float = 5.0) -> tuple[np.ndarray, np.ndarray]:
    rest = np.zeros((n, 3), dtype=np.float32)
    rest[:, 2] = np.arange(n, dtype=np.float32) * spacing
    rest_bone_lengths = np.full(n - 1, spacing, dtype=np.float32)
    return rest, rest_bone_lengths


def test_rest_pose_fidelity_when_dlc_equals_rest() -> None:
    """All-confident DLC = rest -> output ≈ rest within 1 mm RMS."""
    rest, bones = _rest_chain()
    dlc = rest.copy()
    conf = np.ones(rest.shape[0], dtype=np.float32)
    out = compute_anatomical_targets(dlc, rest, conf, bones)
    rms = float(np.sqrt(np.mean((out - rest) ** 2)))
    assert rms < 1.0, f"rest-pose RMS error {rms:.3f} mm — should be ~0"


def test_arc_length_conservation_clamps_overstretch() -> None:
    """DLC predicts +50% chain stretch -> output arc length is at most
    1.08 * rest arc length."""
    rest, bones = _rest_chain()
    L_rest = float(np.sum(bones))
    # Stretch every segment by 1.5x along Z (no rotation, just elongation)
    dlc = rest.copy()
    dlc[:, 2] = rest[:, 2] * 1.5
    conf = np.ones(rest.shape[0], dtype=np.float32)
    # Disable per-bone stiffness so the test isolates arc-length conservation.
    params = AnatomicalTargetParams(
        stiffness_root=0.0,
        stiffness_tip=0.0,
        max_displacement_root_mm=1e6,
        max_displacement_tip_mm=1e6,
    )
    out = compute_anatomical_targets(dlc, rest, conf, bones, params=params)
    L_out = float(np.sum(np.linalg.norm(np.diff(out, axis=0), axis=1)))
    assert L_out <= 1.08 * L_rest + 1e-3, (
        f"arc length {L_out:.2f} mm exceeds 1.08·{L_rest:.2f}"
    )
    # Also check it's not artificially under-shrunk.
    assert L_out >= 1.07 * L_rest - 1e-3, (
        f"arc length {L_out:.2f} mm under 1.07·{L_rest:.2f} — over-clamped"
    )


def test_arc_length_conservation_clamps_compression() -> None:
    """DLC predicts -30% compression -> output arc length is at least
    0.92 * rest arc length."""
    rest, bones = _rest_chain()
    L_rest = float(np.sum(bones))
    dlc = rest.copy()
    dlc[:, 2] = rest[:, 2] * 0.7
    conf = np.ones(rest.shape[0], dtype=np.float32)
    params = AnatomicalTargetParams(
        stiffness_root=0.0,
        stiffness_tip=0.0,
        max_displacement_root_mm=1e6,
        max_displacement_tip_mm=1e6,
    )
    out = compute_anatomical_targets(dlc, rest, conf, bones, params=params)
    L_out = float(np.sum(np.linalg.norm(np.diff(out, axis=0), axis=1)))
    assert L_out >= 0.92 * L_rest - 1e-3, (
        f"arc length {L_out:.2f} mm below 0.92·{L_rest:.2f}"
    )


def test_low_confidence_joint_follows_neighbours_not_rest() -> None:
    """A single keypoint with conf=0.05 (below threshold 0.1) should NOT
    snap back to rest when both neighbours are high-confidence and
    displaced consistently. Instead it follows their drift."""
    rest, bones = _rest_chain()
    dlc = rest.copy()
    # Shift the entire chain 10 mm in +Y (consistent drift).
    dlc[:, 1] += 10.0
    conf = np.ones(rest.shape[0], dtype=np.float32)
    # Joint 5 is "unreliable"
    conf[5] = 0.05

    # Disable rigidity + arc-length conservation so this test isolates
    # the confidence-weighted blend.
    params = AnatomicalTargetParams(
        stiffness_root=0.0,
        stiffness_tip=0.0,
        max_displacement_root_mm=1e6,
        max_displacement_tip_mm=1e6,
        arc_length_min_ratio=0.0,
        arc_length_max_ratio=1e6,
    )
    out = compute_anatomical_targets(dlc, rest, conf, bones, params=params)
    # Joint 5 should be ~10 mm above rest (followed neighbour drift),
    # not 0 mm above rest.
    drift_y = float(out[5, 1] - rest[5, 1])
    assert drift_y > 5.0, (
        f"low-conf joint 5 drifted only {drift_y:.2f} mm in Y — expected ≈ 10 "
        "mm following its neighbours"
    )


def test_per_bone_rigidity_per_joint_clamps_independently() -> None:
    """Joint 0 (root, max_disp 8) and joint 10 (tip, max_disp 25) clamp
    differently for the same input."""
    rest, _ = _rest_chain()
    targets = rest.copy()
    targets[:, 1] = 50.0  # everyone wants to move 50 mm in +Y
    n = rest.shape[0]
    stiffness = np.zeros(n, dtype=np.float32)
    max_disp = np.linspace(8.0, 25.0, n, dtype=np.float32)
    out = per_bone_rigidity(targets, rest, stiffness, max_disp)
    d_root = float(np.linalg.norm(out[0] - rest[0]))
    d_tip = float(np.linalg.norm(out[-1] - rest[-1]))
    assert d_root <= 8.0 + 1e-3, f"root displacement {d_root:.2f} > 8 mm"
    assert d_tip <= 25.0 + 1e-3, f"tip displacement {d_tip:.2f} > 25 mm"
    # And the tip should be allowed FURTHER than the root.
    assert d_tip > d_root + 5.0, (
        f"tip {d_tip:.2f} should be ≥ root {d_root:.2f} + 5 mm "
        "(per-bone ramp not active)"
    )


def test_confidence_weighted_blend_high_conf_passthrough() -> None:
    """All-confident input passes through (no neighbour blending)."""
    rest, _ = _rest_chain()
    dlc = rest + np.array([0, 5, 0], dtype=np.float32)
    conf = np.ones(rest.shape[0], dtype=np.float32)
    out = confidence_weighted_blend(dlc, rest, conf)
    assert np.allclose(out, dlc, atol=1e-4)


def test_arc_length_conserve_no_op_inside_range() -> None:
    """Targets inside the allowed ratio are returned unchanged."""
    rest, bones = _rest_chain()
    targets = rest.copy()
    out = arc_length_conserve(targets, bones, min_ratio=0.92, max_ratio=1.08)
    assert np.allclose(out, targets)
