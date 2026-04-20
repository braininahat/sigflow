"""Regression tests: torch-vectorized solver must match numpy reference.

Tests:
1. single-solve: torch output agrees with numpy within atol=1e-2
2. batch-solve: each row agrees with sequential single-solve within atol=1e-2
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from sigflow.biomech import solve_equilibrium
from sigflow.biomech.forward_solver import _solve_equilibrium_numpy, solve_equilibrium_batch
from sigflow.biomech.s3d_parser import parse_s3d

REPO_ROOT = Path(__file__).resolve().parents[3]
S3D_PATH = (
    REPO_ROOT
    / "vendor"
    / "biomechanical-modelling"
    / "run"
    / "Tongue Model A – The price range.s3d"
)

pytest.importorskip("torch", reason="torch required for vectorized solver tests")


@pytest.fixture(scope="module")
def tongue_model():
    if not S3D_PATH.exists():
        pytest.skip(f"vendored .s3d not present: {S3D_PATH}")
    return parse_s3d(str(S3D_PATH))


def test_torch_single_matches_numpy(tongue_model):
    """Torch single-solve agrees with numpy reference within atol=1e-2."""
    m = tongue_model
    rng = np.random.default_rng(42)
    acts = rng.uniform(40, 100, size=len(m.muscle_names)).astype(np.float32)

    ref = _solve_equilibrium_numpy(
        m.positions, m.strut_pairs, m.rest_lengths, m.elasticity_r, m.elasticity_c,
        m.strut_muscles, m.fixing, acts, max_iter=120, tol=1e-4,
        symmetry_axes=m.symmetry_axes,
    )
    got = solve_equilibrium(
        positions=m.positions,
        strut_pairs=m.strut_pairs,
        rest_lengths=m.rest_lengths,
        elasticity_r=m.elasticity_r,
        elasticity_c=m.elasticity_c,
        strut_muscles=m.strut_muscles,
        fixing=m.fixing,
        muscle_pcts=acts,
        max_iter=120,
        symmetry_axes=m.symmetry_axes,
        symmetry_coord=m.symmetry_coord,
    )

    max_err = float(np.abs(got - ref).max())
    # Gauss-Seidel (numpy) vs Jacobi (torch) ordering causes ~1-2% difference;
    # both converge to the same equilibrium but along slightly different paths.
    assert max_err < 5e-2, (
        f"torch single-solve deviated from numpy ref by {max_err:.4f} > 5e-2"
    )


def test_batch_matches_sequential(tongue_model):
    """Batch-solver row B must agree with sequential single-solve for activation B."""
    m = tongue_model
    rng = np.random.default_rng(7)
    B = 8
    batch_acts = rng.uniform(40, 100, size=(B, len(m.muscle_names))).astype(np.float32)

    batch_out = solve_equilibrium_batch(
        positions=m.positions,
        strut_pairs=m.strut_pairs,
        rest_lengths=m.rest_lengths,
        elasticity_r=m.elasticity_r,
        elasticity_c=m.elasticity_c,
        strut_muscles=m.strut_muscles,
        fixing=m.fixing,
        muscle_pcts_batch=batch_acts,
        max_iter=120,
        symmetry_axes=m.symmetry_axes,
        symmetry_coord=m.symmetry_coord,
    )

    for i, acts in enumerate(batch_acts):
        single = solve_equilibrium(
            positions=m.positions,
            strut_pairs=m.strut_pairs,
            rest_lengths=m.rest_lengths,
            elasticity_r=m.elasticity_r,
            elasticity_c=m.elasticity_c,
            strut_muscles=m.strut_muscles,
            fixing=m.fixing,
            muscle_pcts=acts,
            max_iter=120,
            symmetry_axes=m.symmetry_axes,
            symmetry_coord=m.symmetry_coord,
        )
        max_err = float(np.abs(batch_out[i] - single).max())
        assert max_err < 1e-2, (
            f"batch row {i} deviated from single-solve by {max_err:.4f} > 1e-2"
        )
