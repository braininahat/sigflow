"""Forward-solver symmetry handling must not collapse the mesh.

Originally written for the sigflow-convention solver (0 = relaxed,
100 = contracted). After Addendum 6's MyoSim3D port, the solver uses
MyoSim3D convention (100 = rest, 0 = fully contracted), so the
rest-pose invariant now requires `muscle_pcts = 100`.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from sigflow.biomech import solve_equilibrium
from sigflow.biomech.s3d_parser import parse_s3d

REPO_ROOT = Path(__file__).resolve().parents[3]  # .../ultraspeech-pyqt
S3D_PATH = (
    REPO_ROOT
    / "vendor"
    / "biomechanical-modelling"
    / "run"
    / "Tongue Model A – The price range.s3d"
)


@pytest.fixture(scope="module")
def tongue_model():
    if not S3D_PATH.exists():
        pytest.skip(f"vendored .s3d not present: {S3D_PATH}")
    return parse_s3d(str(S3D_PATH))


def test_zero_activation_preserves_rest_pose(tongue_model):
    """At muscle_pcts=0 the solver should leave .s3d rest positions alone.

    Pre-fix: the symmetry code damped X by 0.5 every iteration, so after
    N iters X collapsed by 2**-N → atoms pancaked onto the midsagittal
    plane.
    Post-fix: only midline atoms get pinned to the symmetry plane;
    off-midline atoms stay put.
    """
    model = tongue_model
    solved = solve_equilibrium(
        positions=model.positions,
        strut_pairs=model.strut_pairs,
        rest_lengths=model.rest_lengths,
        elasticity_r=model.elasticity_r,
        elasticity_c=model.elasticity_c,
        strut_muscles=model.strut_muscles,
        fixing=model.fixing_enum,
        muscle_pcts=np.full(len(model.muscle_names), 100.0, dtype=np.float32),
        max_iter=80,
        symmetry_axes=model.symmetry_axes,
        symmetry_coord=model.symmetry_coord,
    )

    # Off-midline atoms (|X| > small threshold) must not have moved.
    axis_range = float(model.positions[:, 0].max() - model.positions[:, 0].min())
    threshold = 0.01 * axis_range
    off_midline = np.abs(model.positions[:, 0]) >= threshold

    disp = np.linalg.norm(solved - model.positions, axis=1)
    max_off_midline_disp = float(disp[off_midline].max())

    # Tolerance of 1.0 absorbs the small spring-equilibrium jitter from
    # .s3d rest lengths not exactly matching authored atom distances
    # (max residual ~1.65 units per strut). Pre-fix this test saw
    # displacements in the 100s.
    assert max_off_midline_disp < 1.0, (
        f"off-midline atoms moved at muscle_pcts=0 "
        f"(max disp = {max_off_midline_disp:.4f}); symmetry enforcement "
        f"is still damping the mesh."
    )


def test_zero_activation_pins_midline(tongue_model):
    """chSym atoms (symmetry-plane) must not move along the symmetry axis.

    MyoSim3D's chSym atoms sit on the symmetry plane (X = -0.32 in this
    model per the `Y:` record). Their X coordinate must remain constant
    across solves, regardless of activation.
    """
    model = tongue_model
    solved = solve_equilibrium(
        positions=model.positions,
        strut_pairs=model.strut_pairs,
        rest_lengths=model.rest_lengths,
        elasticity_r=model.elasticity_r,
        elasticity_c=model.elasticity_c,
        strut_muscles=model.strut_muscles,
        fixing=model.fixing_enum,
        muscle_pcts=np.full(len(model.muscle_names), 50.0, dtype=np.float32),
        max_iter=40,
        symmetry_axes=(0,),
        symmetry_coord=model.symmetry_coord,
    )

    sym_mask = model.fixing_enum == 2
    if not sym_mask.any():
        pytest.skip("no chSym atoms in this model")

    # X drift along the symmetry axis must be exactly zero for chSym atoms.
    max_x_drift = float(np.abs(solved[sym_mask, 0] - model.positions[sym_mask, 0]).max())
    assert max_x_drift < 1e-6, (
        f"chSym atoms drifted along symmetry axis (max |Δx| = {max_x_drift})"
    )


def test_no_symmetry_axes_leaves_positions_alone(tongue_model):
    """With no symmetry enforcement + zero activations, solver is a no-op."""
    model = tongue_model
    solved = solve_equilibrium(
        positions=model.positions,
        strut_pairs=model.strut_pairs,
        rest_lengths=model.rest_lengths,
        elasticity_r=model.elasticity_r,
        elasticity_c=model.elasticity_c,
        strut_muscles=model.strut_muscles,
        fixing=model.fixing_enum,
        muscle_pcts=np.full(len(model.muscle_names), 100.0, dtype=np.float32),
        max_iter=30,
        symmetry_axes=(),
        symmetry_coord=model.symmetry_coord,
    )
    # 1.0 tolerance: residual spring jitter since .s3d rest lengths
    # don't exactly match authored atom distances.
    max_disp = float(np.linalg.norm(solved - model.positions, axis=1).max())
    assert max_disp < 1.0, f"solver moved atoms at 0-act with no-sym; max={max_disp}"
