"""Faithful Python port of MyoSim3D's CalcOneIntervalSolve.

Source: vendor/biomechanical-modelling/src/Unit1.pas lines 5899-6502.

This is a reference implementation — not vectorised, not fast. Its job
is to be a line-by-line transliteration of the Delphi algorithm so the
vectorised PyTorch port can be tested against it for numerical parity.

Algorithm summary (from Unit1.pas):

  For each pass in range(max_iter):
    # 1. Per-strut target length and blended elasticity
    for il in 0 .. N_struts-1:
      R = rest_lengths[il]
      if strut_muscles[il] >= 0:
        TargLen[il] = max(R * muscle_pcts[strut_muscles[il]] / 100, 1.0)
        Elasticity[il] = (ElasticityR[il] - ElasticityC[il]) * TargLen[il]/R
                         + ElasticityC[il]
      else:
        TargLen[il] = max(R, 1.0)
        Elasticity[il] = ElasticityR[il]

    # 2. Zero per-atom forces and per-atom weight sums
    atom_f[:] = 0
    wgt[:] = 0

    # 3. Spring forces with DEADBAND (only if stretched past target)
    for il in 0 .. N_struts-1:
      a1, a2 = strut_pairs[il]
      vect = atom_p[a1] - atom_p[a2]
      d1 = ||vect||
      if d1 > 0:
        b = (d1 - TargLen[il]) / d1
        if b > 0:                      # DEADBAND
          b = b / 2
          vect = vect * b
          atom_f[a1] -= vect * Elasticity[il]; wgt[a1] += Elasticity[il]
          atom_f[a2] += vect * Elasticity[il]; wgt[a2] += Elasticity[il]

    # 4. Normalise force by per-atom stiffness sum (weight)
    for ia:
      if wgt[ia] != 0:
        atom_f[ia] /= wgt[ia]

    # 5. Volume pressure (AddPressureBrick) — deferred to v2; see TODO

    # 6. Apply forces to positions with symmetry handling
    for ia:
      if fixing[ia] == FREE:
        atom_p[ia] += atom_f[ia]
        # optional symmetry-plane clamp (omitted for tongue model)
      elif fixing[ia] == SYM:
        # midline atom — move only in the 2 axes perpendicular to
        # the symmetry axis
        atom_p[ia, non_sym_axes] += atom_f[ia, non_sym_axes]
      # STATIC: no update

    # 7. ApplyConstraints, ApplyRigity — no-ops in our case (no roof, no rigid bodies)

User-convention wrapper
-----------------------

MyoSim3D convention: muscle slider at 0   = fully contracted (TargLen → 0, clamped to 1).
                     muscle slider at 100 = rest (TargLen = RestLength).

Our demo's user convention: 100 = rest, 0 = full contraction — same as MyoSim3D.
(See biomech_demo.py._to_solver_pcts.)

However sigflow's existing convention (before this port): 0 = relaxed,
100 = contracted. Callers that pass sigflow-convention must invert
via `100 - pct` before calling.
"""
from __future__ import annotations

import numpy as np

from .types import MyoSim3D


FIXING_FREE = 0
FIXING_STATIC = 1
FIXING_SYM = 2


def solve_equilibrium_reference(
    model: MyoSim3D,
    muscle_pcts: np.ndarray,
    max_iter: int = 100,
    symmetry_axis: int = 0,
) -> np.ndarray:
    """Run MyoSim3D's static Jacobi + deadband solver.

    Args:
        model: Parsed MyoSim3D tongue model.
        muscle_pcts: (n_muscles,) in MyoSim3D convention (0 = contracted,
            100 = rest). Length must cover the max muscle ID + 1.
        max_iter: Solver iterations. MyoSim3D default = 100 via
            dlgOptions.seSolve.value.
        symmetry_axis: 0 = X, 1 = Y, 2 = Z. For the tongue model,
            symmetry is across X = 0, so axis = 0.

    Returns:
        (n_atoms, 3) float32 array of deformed positions.
    """
    pos = model.positions.astype(np.float64).copy()
    strut_pairs = model.strut_pairs
    rest_len = model.rest_lengths.astype(np.float64)
    elas_r = model.elasticity_r.astype(np.float64)
    elas_c = model.elasticity_c.astype(np.float64)
    strut_musc = model.strut_muscles
    if model.fixing_enum is not None:
        fixing_int = np.asarray(model.fixing_enum, dtype=np.int8)
    else:
        # Fallback: derive a coarse enum from legacy bool `fixing`
        # (treats all bool-True atoms as chStatic, ignores chSym)
        fixing_int = np.asarray(model.fixing, dtype=np.int8)

    n_atoms = pos.shape[0]
    n_struts = strut_pairs.shape[0]
    muscle_pcts = np.asarray(muscle_pcts, dtype=np.float64)

    # Precompute per-strut target length and elasticity (constant across iterations)
    targ_len = np.empty(n_struts, dtype=np.float64)
    elasticity = np.empty(n_struts, dtype=np.float64)
    for il in range(n_struts):
        R = rest_len[il]
        m = int(strut_musc[il])
        if m >= 0:
            pct = muscle_pcts[m] if m < len(muscle_pcts) else 100.0
            tl = R * pct / 100.0
            targ_len[il] = max(tl, 1.0)
            e = (elas_r[il] - elas_c[il]) * targ_len[il] / R + elas_c[il]
            elasticity[il] = e
        else:
            targ_len[il] = max(R, 1.0)
            elasticity[il] = elas_r[il]

    # Which axes can a sym atom move in? All except `symmetry_axis`.
    sym_free_axes = [i for i in range(3) if i != symmetry_axis]

    for _ in range(max_iter):
        atom_f = np.zeros((n_atoms, 3), dtype=np.float64)
        wgt = np.zeros(n_atoms, dtype=np.float64)

        for il in range(n_struts):
            a1, a2 = strut_pairs[il]
            vect = pos[a1] - pos[a2]
            d1 = float(np.sqrt(vect[0] ** 2 + vect[1] ** 2 + vect[2] ** 2))
            if d1 == 0.0:
                continue
            b = (d1 - targ_len[il]) / d1
            if b <= 0.0:
                continue  # DEADBAND: no force when strut is shorter than target
            b = b / 2.0
            vect_scaled = vect * b
            e = elasticity[il]
            atom_f[a1] -= vect_scaled * e
            atom_f[a2] += vect_scaled * e
            wgt[a1] += e
            wgt[a2] += e

        nz = wgt != 0.0
        atom_f[nz] /= wgt[nz, None]

        # 5. (volume pressure not yet ported — deferred to v2)

        for ia in range(n_atoms):
            fx = int(fixing_int[ia])
            if fx == FIXING_FREE:
                pos[ia] += atom_f[ia]
            elif fx == FIXING_SYM:
                for ax in sym_free_axes:
                    pos[ia, ax] += atom_f[ia, ax]
            # FIXING_STATIC: no movement

    return pos.astype(np.float32)
