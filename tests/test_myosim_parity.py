"""Sanity tests for the MyoSim3D-faithful solver port (Addendum 6).

Full bit-parity against the Delphi reference is not yet possible — that
needs CSV exports from a Linux-native MyoSim3D build (deferred task).
These tests cover what we can check now:

  - vectorised `solve_equilibrium` matches the NumPy reference impl
  - `_myosim_reference.solve_equilibrium_reference` matches both
  - stability: no NaN/Inf under extreme activations
  - deformation bounded by 2× model bbox diagonal
  - batched solve matches single solves
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from sigflow.biomech._myosim_reference import solve_equilibrium_reference
from sigflow.biomech.forward_solver import (
    _solve_equilibrium_numpy,
    solve_equilibrium,
    solve_equilibrium_batch,
)
from sigflow.biomech.s3d_parser import parse_s3d


REPO_ROOT = Path(__file__).resolve().parents[3]
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


@pytest.fixture(scope="module")
def activation_cases(tongue_model):
    """Representative activation vectors for stability coverage."""
    n_m = len(tongue_model.muscle_names)
    rng = np.random.default_rng(42)
    return [
        ("rest", np.full(n_m, 100.0, dtype=np.float32)),
        ("half", np.full(n_m, 50.0, dtype=np.float32)),
        ("max_contract", np.zeros(n_m, dtype=np.float32)),
        ("single_m5", np.where(np.arange(n_m) == 5, 20.0, 100.0).astype(np.float32)),
        ("random_mid", rng.uniform(40, 100, size=n_m).astype(np.float32)),
        ("random_full", rng.uniform(0, 100, size=n_m).astype(np.float32)),
    ]


def _solve(model, pcts, **kw):
    return solve_equilibrium(
        positions=model.positions,
        strut_pairs=model.strut_pairs,
        rest_lengths=model.rest_lengths,
        elasticity_r=model.elasticity_r,
        elasticity_c=model.elasticity_c,
        strut_muscles=model.strut_muscles,
        fixing=model.fixing_enum,
        muscle_pcts=pcts,
        symmetry_axes=(0,),
        **kw,
    )


def test_all_cases_finite_and_bounded(tongue_model, activation_cases):
    """No activation vector should produce NaN/Inf or leave the bbox × 2."""
    bbox_diag = float(
        np.linalg.norm(tongue_model.positions.max(0) - tongue_model.positions.min(0))
    )
    bound = 2.0 * bbox_diag

    for tag, pcts in activation_cases:
        pos = _solve(tongue_model, pcts, max_iter=50)
        assert np.all(np.isfinite(pos)), f"{tag}: NaN/Inf"
        assert float(np.abs(pos).max()) < bound, (
            f"{tag}: atom escaped (max |pos|={np.abs(pos).max():.1f} > {bound:.1f})"
        )


def test_rest_is_near_fixed_point(tongue_model):
    """At muscle_pcts = 100 (MyoSim3D rest), the solver should barely move atoms."""
    n_m = len(tongue_model.muscle_names)
    pos = _solve(tongue_model, np.full(n_m, 100.0, dtype=np.float32), max_iter=50)
    drift = float(np.abs(pos - tongue_model.positions).max())
    assert drift < 1.0, f"rest-pose drift too large: {drift}"


def test_vectorised_matches_numpy_reference(tongue_model):
    """Torch vectorised path ≈ numpy reference path."""
    n_m = len(tongue_model.muscle_names)
    pcts = np.full(n_m, 60.0, dtype=np.float32)
    max_iter = 20

    pos_vec = _solve(tongue_model, pcts, max_iter=max_iter)
    pos_np = _solve_equilibrium_numpy(
        positions=tongue_model.positions,
        strut_pairs=tongue_model.strut_pairs,
        rest_lengths=tongue_model.rest_lengths,
        elasticity_r=tongue_model.elasticity_r,
        elasticity_c=tongue_model.elasticity_c,
        strut_muscles=tongue_model.strut_muscles,
        fixing=tongue_model.fixing_enum,
        muscle_pcts=pcts,
        max_iter=max_iter,
        tol=1e-4,
        symmetry_axes=(0,),
    )
    err = float(np.abs(pos_vec - pos_np).max())
    # 1.0 mm tol: float32 accumulated over 20 iters on a 1158-atom mesh
    assert err < 1.0, f"vectorised vs numpy-ref max error = {err}"


def test_reference_matches_forward_solver(tongue_model):
    """_myosim_reference.solve_equilibrium_reference ≈ production solver."""
    n_m = len(tongue_model.muscle_names)
    pcts = np.full(n_m, 60.0, dtype=np.float32)
    max_iter = 20

    pos_ref = solve_equilibrium_reference(tongue_model, pcts, max_iter=max_iter)
    pos_prod = _solve(tongue_model, pcts, max_iter=max_iter)
    err = float(np.abs(pos_ref - pos_prod).max())
    assert err < 1.0, f"reference vs production max error = {err}"


def test_batch_matches_single(tongue_model, activation_cases):
    """Batched solve must match per-case singles (within float32 noise)."""
    cases = activation_cases[:3]
    pcts_batch = np.stack([c[1] for c in cases]).astype(np.float32)
    max_iter = 20

    pos_batch = solve_equilibrium_batch(
        positions=tongue_model.positions,
        strut_pairs=tongue_model.strut_pairs,
        rest_lengths=tongue_model.rest_lengths,
        elasticity_r=tongue_model.elasticity_r,
        elasticity_c=tongue_model.elasticity_c,
        strut_muscles=tongue_model.strut_muscles,
        fixing=tongue_model.fixing_enum,
        muscle_pcts_batch=pcts_batch,
        max_iter=max_iter,
        symmetry_axes=(0,),
    )
    for i, (tag, pcts) in enumerate(cases):
        pos_single = _solve(tongue_model, pcts, max_iter=max_iter)
        err = float(np.abs(pos_batch[i] - pos_single).max())
        assert err < 1.0, f"[{tag}] batch vs single max error = {err}"


def test_step_matches_single_relaxation(tongue_model):
    """BiomechStepGNO's forward must output a finite, shape-correct Δpose.

    Untrained-weights smoke test. A trained-network fidelity check
    (learned step ≈ force-balance step within some RMSE) belongs in an
    integration test that runs live training for N seconds — out of
    scope for fast unit tests. This test only verifies the plumbing:

    - input/output shapes match the force-balance rollout
    - output is finite
    - one step from rest with random-init weights doesn't produce NaN
    """
    try:
        import torch
        from sigflow.biomech import BiomechStepGNO, solve_equilibrium_rollout
    except Exception as e:
        pytest.skip(f"torch / GINO stack unavailable: {e}")

    model = tongue_model
    n_m = len(model.muscle_names)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    gno = BiomechStepGNO(model).to(device)
    gno.eval()

    pcts = np.full(n_m, 70.0, dtype=np.float32)
    traj = solve_equilibrium_rollout(
        positions=model.positions,
        strut_pairs=model.strut_pairs,
        rest_lengths=model.rest_lengths,
        elasticity_r=model.elasticity_r,
        elasticity_c=model.elasticity_c,
        strut_muscles=model.strut_muscles,
        fixing=model.fixing_enum,
        muscle_pcts=pcts,
        max_iter=3,
        symmetry_axes=(0,),
    )
    assert traj.shape[0] >= 2, "rollout must produce at least rest + one step"

    pose0 = torch.from_numpy(traj[0]).unsqueeze(0).to(device)
    acts = torch.from_numpy(pcts).unsqueeze(0).to(device)
    with torch.no_grad():
        dpose = gno(pose0, acts)
    assert tuple(dpose.shape) == (1, model.positions.shape[0], 3), (
        f"unexpected step output shape {tuple(dpose.shape)}"
    )
    assert bool(torch.isfinite(dpose).all()), "BiomechStepGNO output has NaN/Inf"


def test_rollout_shape_and_finite(tongue_model):
    """rollout() should return a terminal pose of the right shape.

    Batch > 1 is intentionally not tested — neuralop's GINO neighbor
    search assumes batch=1 for shared geometry in this version
    (known limitation; raises in layers/neighbor_search.py). Demo and
    InferenceWorker always call with batch=1.
    """
    try:
        import torch
        from sigflow.biomech import BiomechStepGNO
    except Exception as e:
        pytest.skip(f"torch / GINO stack unavailable: {e}")

    model = tongue_model
    n_m = len(model.muscle_names)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    gno = BiomechStepGNO(model).to(device)
    gno.eval()
    acts = torch.full((1, n_m), 100.0, device=device)
    with torch.no_grad():
        terminal = gno.rollout(acts, max_iter=5)
    assert tuple(terminal.shape) == (1, model.positions.shape[0], 3)
    assert bool(torch.isfinite(terminal).all())


def test_deadband_passive_at_rest(tongue_model):
    """Passive struts (strut_muscles == -1) have TargLen = RestLength.

    At rest, their current length approximately equals target (within
    .s3d authoring jitter), so the deadband should suppress most of
    their forces and the solver should barely move atoms incident only
    to passive struts.
    """
    n_m = len(tongue_model.muscle_names)
    pos_rest_100 = _solve(tongue_model, np.full(n_m, 100.0, dtype=np.float32), max_iter=30)
    drift = np.linalg.norm(pos_rest_100 - tongue_model.positions, axis=1).max()
    # Small non-zero drift is acceptable (authoring jitter in rest lengths).
    assert drift < 2.0, f"rest-pose drift too large: {drift}"
