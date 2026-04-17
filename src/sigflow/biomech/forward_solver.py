"""Biomechanical forward solver — muscle activations → tongue shape.

Port of MyoSim3D's static Jacobi + deadband solver
(vendor/biomechanical-modelling/src/Unit1.pas:5899-6502).

Algorithm (per iteration):
  1. Per-strut target length and blended elasticity from activation.
     TargLen = max(rest * pct/100, 1.0)            [Unit1.pas:6277]
     Elasticity = (ElasticityR - ElasticityC) * TargLen/rest + ElasticityC
                  for muscle struts; else ElasticityR  [Unit1.pas:6279,6284]
  2. Zero per-atom force and weight accumulators.
  3. For each strut:
       vect = pos[a1] - pos[a2]; d1 = ||vect||
       b = (d1 - TargLen)/d1
       if b > 0:  # DEADBAND — no force on compressed struts
         vect *= b/2
         f[a1] -= vect*Elasticity; wgt[a1] += Elasticity
         f[a2] += vect*Elasticity; wgt[a2] += Elasticity
  4. Normalise: f /= wgt (per atom with wgt > 0)
  5. Apply update:
       chFree atoms: pos += f
       chSym atoms:  pos[axes != symmetry_axis] += f[...]  (midline-constrained)
       chStatic atoms: no movement

Two implementations:
- torch-backed (CPU or CUDA): vectorised via scatter_add. Production.
- numpy fallback: strut-by-strut Python loop. Reference impl for tests.

Convention note: `muscle_pcts` is in MyoSim3D slider convention
  0   = fully contracted (target length clamps to 1.0)
  100 = rest              (target length == RestLength)
which matches the biomech demo's user-facing slider. sigflow-convention
callers (0=relaxed, 100=contracted) must invert at their boundary.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


# TFixing enum values from Unit1.pas:159 (TFixing = (chFree, chStatic, chSym))
_FIXING_FREE = 0
_FIXING_SYM = 2  # chStatic = 1 is implicit: "neither free nor sym"


def _get_device(device: Optional[str] = None):
    """Return a torch device, preferring CUDA when available."""
    import torch

    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _fixing_to_enum(fixing: np.ndarray) -> np.ndarray:
    """Coerce legacy bool fixing arrays to the tri-state enum.

    Bool-True maps to chStatic (the old single-flag meaning).
    For full symmetry-plane support, callers should pass `fixing_enum`
    directly via the `fixing` arg.
    """
    arr = np.asarray(fixing)
    if arr.dtype == bool:
        return arr.astype(np.int8)  # True=1=chStatic, False=0=chFree
    return arr.astype(np.int8)


def _compute_targ_and_elasticity(
    rest_lengths: np.ndarray,
    elasticity_r: np.ndarray,
    elasticity_c: np.ndarray,
    strut_muscles: np.ndarray,
    muscle_pcts: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Port of Unit1.pas:6271-6285 — per-strut TargLen + blended Elasticity.

    Returns (targ_len, elasticity), both (N_struts,) float32.
    """
    rest = np.asarray(rest_lengths, dtype=np.float32)
    er = np.asarray(elasticity_r, dtype=np.float32)
    ec = np.asarray(elasticity_c, dtype=np.float32)
    sm = np.asarray(strut_muscles, dtype=np.int64)
    pcts = np.clip(np.asarray(muscle_pcts, dtype=np.float32), 0.0, 100.0)

    targ = rest.copy()
    elas = er.copy()

    active = sm >= 0
    if active.any():
        # muscle-driven struts: TargLen = max(rest * pct/100, 1)
        pct_per_strut = pcts[sm[active]]
        targ_active = np.maximum(rest[active] * pct_per_strut / 100.0, 1.0)
        targ[active] = targ_active
        # elasticity blends between ElasticityR and ElasticityC by TargLen/rest
        elas_active = (
            (er[active] - ec[active]) * (targ_active / rest[active]) + ec[active]
        )
        elas[active] = elas_active
    # passive struts: TargLen = max(rest, 1), Elasticity = ElasticityR (already set)
    passive = ~active
    if passive.any():
        targ[passive] = np.maximum(rest[passive], 1.0)

    return targ, elas


def _solve_equilibrium_torch(
    positions: np.ndarray,
    strut_pairs: np.ndarray,
    rest_lengths: np.ndarray,
    elasticity_r: np.ndarray,
    elasticity_c: np.ndarray,
    strut_muscles: np.ndarray,
    fixing: np.ndarray,
    muscle_pcts: np.ndarray,
    max_iter: int,
    tol: float,
    symmetry_axes: tuple[int, ...],
    device,
    return_trajectory: bool = False,
) -> np.ndarray:
    import torch

    n_atoms = len(positions)
    pos = torch.tensor(positions, dtype=torch.float32, device=device)
    trajectory = [pos.cpu().numpy().copy()] if return_trajectory else None
    a_idx = torch.tensor(strut_pairs[:, 0], dtype=torch.long, device=device)
    b_idx = torch.tensor(strut_pairs[:, 1], dtype=torch.long, device=device)

    targ_np, elas_np = _compute_targ_and_elasticity(
        rest_lengths, elasticity_r, elasticity_c, strut_muscles, muscle_pcts
    )
    targ = torch.tensor(targ_np, dtype=torch.float32, device=device)
    elas = torch.tensor(elas_np, dtype=torch.float32, device=device)

    fixing_enum = _fixing_to_enum(fixing)
    fix_t = torch.tensor(fixing_enum, dtype=torch.int8, device=device)
    is_free = fix_t == _FIXING_FREE
    is_sym = fix_t == _FIXING_SYM
    # STATIC atoms are implicitly excluded (neither free nor sym)

    # Sym-atom movement masks: True on axes where sym atoms ARE allowed to move.
    # For a symmetry plane on axis S, sym atoms move freely on the other two.
    sym_axis = symmetry_axes[0] if symmetry_axes else 0
    free_axes = torch.ones(3, dtype=torch.bool, device=device)
    free_axes[sym_axis] = False  # sym atoms cannot move along the symmetry axis

    a_exp = a_idx.unsqueeze(-1).expand(-1, 3)
    b_exp = b_idx.unsqueeze(-1).expand(-1, 3)

    eps = 1e-8

    for _ in range(max_iter):
        pos_old = pos.clone()

        # Spring force per strut (deadband: only b > 0)
        vect = pos[a_idx] - pos[b_idx]                    # (N_struts, 3)
        d1 = vect.norm(dim=1)                             # (N_struts,)
        safe_d1 = d1.clamp(min=eps)
        b = (d1 - targ) / safe_d1                        # (N_struts,)
        active = b > 0
        b_half = torch.where(active, b / 2.0, torch.zeros_like(b))  # (N_struts,)

        f_vec = vect * (b_half * elas).unsqueeze(-1)      # (N_struts, 3)

        force = torch.zeros(n_atoms, 3, dtype=torch.float32, device=device)
        force.scatter_add_(0, a_exp, -f_vec)              # note: f_a = -vect*b/2*elas
        force.scatter_add_(0, b_exp, +f_vec)

        # Per-atom weight = sum of Elasticity over active incident struts
        wgt = torch.zeros(n_atoms, dtype=torch.float32, device=device)
        active_elas = torch.where(active, elas, torch.zeros_like(elas))
        wgt.index_add_(0, a_idx, active_elas)
        wgt.index_add_(0, b_idx, active_elas)
        wgt_safe = wgt.clamp(min=eps).unsqueeze(-1)

        update = force / wgt_safe                         # (N_atoms, 3)
        # Zero out atoms with zero weight (no active incident strut = no net force)
        update = torch.where(
            wgt.unsqueeze(-1) > 0, update, torch.zeros_like(update)
        )

        # Apply by fixing kind
        #   FREE: full update
        #   SYM : update only on non-symmetry axes
        #   STATIC: no update
        new_pos = pos.clone()
        new_pos[is_free] = pos[is_free] + update[is_free]
        if is_sym.any():
            sym_update = update[is_sym] * free_axes.unsqueeze(0)
            new_pos[is_sym] = pos[is_sym] + sym_update
        pos = new_pos

        if trajectory is not None:
            trajectory.append(pos.cpu().numpy().copy())

        if (pos - pos_old).norm() < tol:
            break

    if trajectory is not None:
        return np.stack(trajectory, axis=0)
    return pos.cpu().numpy()


def _solve_equilibrium_numpy(
    positions: np.ndarray,
    strut_pairs: np.ndarray,
    rest_lengths: np.ndarray,
    elasticity_r: np.ndarray,
    elasticity_c: np.ndarray,
    strut_muscles: np.ndarray,
    fixing: np.ndarray,
    muscle_pcts: np.ndarray,
    max_iter: int,
    tol: float,
    symmetry_axes: tuple[int, ...],
) -> np.ndarray:
    """NumPy reference implementation — matches Unit1.pas strut-by-strut loop."""
    pos = np.asarray(positions, dtype=np.float64).copy()
    n_atoms = len(pos)
    n_struts = len(strut_pairs)

    targ, elas = _compute_targ_and_elasticity(
        rest_lengths, elasticity_r, elasticity_c, strut_muscles, muscle_pcts
    )
    targ = targ.astype(np.float64)
    elas = elas.astype(np.float64)

    fixing_enum = _fixing_to_enum(fixing)
    sym_axis = symmetry_axes[0] if symmetry_axes else 0
    sym_free_axes = [i for i in range(3) if i != sym_axis]

    for _ in range(max_iter):
        pos_old = pos.copy()
        force = np.zeros((n_atoms, 3), dtype=np.float64)
        wgt = np.zeros(n_atoms, dtype=np.float64)

        for il in range(n_struts):
            a1, a2 = strut_pairs[il]
            vect = pos[a1] - pos[a2]
            d1 = float(np.linalg.norm(vect))
            if d1 == 0.0:
                continue
            b = (d1 - targ[il]) / d1
            if b <= 0.0:
                continue
            b /= 2.0
            vect_scaled = vect * b
            e = elas[il]
            force[a1] -= vect_scaled * e
            force[a2] += vect_scaled * e
            wgt[a1] += e
            wgt[a2] += e

        nz = wgt > 0
        force[nz] /= wgt[nz, None]

        for ia in range(n_atoms):
            fx = int(fixing_enum[ia])
            if fx == _FIXING_FREE:
                pos[ia] += force[ia]
            elif fx == _FIXING_SYM:
                for ax in sym_free_axes:
                    pos[ia, ax] += force[ia, ax]

        if np.linalg.norm(pos - pos_old) < tol:
            break

    return pos.astype(np.float32)


def solve_equilibrium_rollout(
    positions: np.ndarray,
    strut_pairs: np.ndarray,
    rest_lengths: np.ndarray,
    elasticity_r: np.ndarray,
    elasticity_c: np.ndarray,
    strut_muscles: np.ndarray,
    fixing: np.ndarray,
    muscle_pcts: np.ndarray,
    max_iter: int = 100,
    tol: float = 1e-4,
    symmetry_axes: tuple[int, ...] = (0,),
    symmetry_coord: int = 1,  # noqa: ARG001 — kept for API compat
) -> np.ndarray:
    """Solve for equilibrium AND return the full per-iteration trajectory.

    Same algorithm as `solve_equilibrium`, but returns every iterate
    from rest to equilibrium for training an iterative surrogate
    (per-step supervision). See Addendum 7 in the biomech plan.

    Returns:
        trajectory: (T+1, N_atoms, 3) float32 where T ≤ max_iter is the
            number of iterations actually taken (early-exit on tol).
            trajectory[0] is the input positions; trajectory[-1] is the
            final equilibrium pose.
    """
    try:
        device = _get_device()
        return _solve_equilibrium_torch(
            positions, strut_pairs, rest_lengths, elasticity_r, elasticity_c,
            strut_muscles, fixing, muscle_pcts, max_iter, tol, symmetry_axes,
            device, return_trajectory=True,
        )
    except Exception:
        # numpy fallback doesn't yet support return_trajectory; reconstruct
        # by iterating max_iter=1 steps. Slow path, only used if torch unavailable.
        traj = [np.asarray(positions, dtype=np.float32).copy()]
        cur = np.asarray(positions, dtype=np.float32).copy()
        for _ in range(max_iter):
            nxt = _solve_equilibrium_numpy(
                cur, strut_pairs, rest_lengths, elasticity_r, elasticity_c,
                strut_muscles, fixing, muscle_pcts, 1, tol, symmetry_axes,
            )
            traj.append(np.asarray(nxt, dtype=np.float32))
            if np.linalg.norm(nxt - cur) < tol:
                break
            cur = nxt
        return np.stack(traj, axis=0)


def solve_equilibrium(
    positions: np.ndarray,
    strut_pairs: np.ndarray,
    rest_lengths: np.ndarray,
    elasticity_r: np.ndarray,
    elasticity_c: np.ndarray,
    strut_muscles: np.ndarray,
    fixing: np.ndarray,
    muscle_pcts: np.ndarray,
    max_iter: int = 100,
    tol: float = 1e-4,
    symmetry_axes: tuple[int, ...] = (0,),
    symmetry_coord: int = 1,  # noqa: ARG001 — kept for API compat
) -> np.ndarray:
    """Solve for equilibrium tongue shape given muscle activations.

    MyoSim3D-faithful port (Unit1.pas:5899).

    Args:
        positions: Rest positions (N_atoms, 3) float32
        strut_pairs: Connectivity (N_struts, 2) int32
        rest_lengths: Rest length per strut (N_struts,) float32
        elasticity_r: Radial elasticity (N_struts,) float32
        elasticity_c: Circumferential elasticity (N_struts,) float32
        strut_muscles: Muscle ID per strut; -1 = passive (N_struts,) int32
        fixing: Fixing per atom — pass `model.fixing_enum` (int8 {0,1,2})
            for full symmetry-plane support, or legacy bool for
            STATIC-only semantics.
        muscle_pcts: Activation per muscle, MyoSim3D convention
            (0 = contracted, 100 = rest). (N_muscles,) float32
        max_iter: Maximum Jacobi iterations (MyoSim3D default 100)
        tol: Convergence tolerance (Frobenius norm of position change)
        symmetry_axes: Axes normal to the symmetry plane (default (0,) = X)
        symmetry_coord: Unused; kept for API compatibility

    Returns:
        Equilibrium positions (N_atoms, 3) float32
    """
    try:
        device = _get_device()
        return _solve_equilibrium_torch(
            positions, strut_pairs, rest_lengths, elasticity_r, elasticity_c,
            strut_muscles, fixing, muscle_pcts, max_iter, tol, symmetry_axes, device,
        )
    except Exception:
        return _solve_equilibrium_numpy(
            positions, strut_pairs, rest_lengths, elasticity_r, elasticity_c,
            strut_muscles, fixing, muscle_pcts, max_iter, tol, symmetry_axes,
        )


def solve_equilibrium_batch(
    positions: np.ndarray,
    strut_pairs: np.ndarray,
    rest_lengths: np.ndarray,
    elasticity_r: np.ndarray,
    elasticity_c: np.ndarray,
    strut_muscles: np.ndarray,
    fixing: np.ndarray,
    muscle_pcts_batch: np.ndarray,
    max_iter: int = 100,
    tol: float = 1e-4,
    symmetry_axes: tuple[int, ...] = (0,),
    symmetry_coord: int = 1,  # noqa: ARG001 — kept for API compat
    device: Optional[str] = None,
) -> np.ndarray:
    """Batched forward solve — all B activation vectors in one GPU pass."""
    import torch

    dev = _get_device(device)
    B = len(muscle_pcts_batch)
    n_atoms = len(positions)

    pos = (
        torch.tensor(positions, dtype=torch.float32, device=dev)
        .unsqueeze(0)
        .expand(B, -1, -1)
        .clone()
    )

    a_idx = torch.tensor(strut_pairs[:, 0], dtype=torch.long, device=dev)
    b_idx = torch.tensor(strut_pairs[:, 1], dtype=torch.long, device=dev)

    # Per-strut TargLen and Elasticity for each batch item
    rest = torch.tensor(rest_lengths, dtype=torch.float32, device=dev)
    er = torch.tensor(elasticity_r, dtype=torch.float32, device=dev)
    ec = torch.tensor(elasticity_c, dtype=torch.float32, device=dev)
    sm = torch.tensor(strut_muscles, dtype=torch.long, device=dev)
    acts = torch.tensor(
        np.clip(muscle_pcts_batch, 0, 100), dtype=torch.float32, device=dev
    )  # (B, N_muscles)

    # Broadcast TargLen: (B, N_struts)
    targ = rest.unsqueeze(0).expand(B, -1).clone()
    elas = er.unsqueeze(0).expand(B, -1).clone()
    active_strut = sm >= 0
    if active_strut.any():
        # acts[:, sm[active]] → (B, n_active_struts)
        pct_per_strut = acts[:, sm[active_strut]]
        targ_active = torch.clamp(
            rest[active_strut].unsqueeze(0) * pct_per_strut / 100.0, min=1.0
        )
        targ[:, active_strut] = targ_active
        er_a = er[active_strut].unsqueeze(0)
        ec_a = ec[active_strut].unsqueeze(0)
        elas[:, active_strut] = (er_a - ec_a) * (
            targ_active / rest[active_strut].unsqueeze(0)
        ) + ec_a
    passive = ~active_strut
    if passive.any():
        targ[:, passive] = torch.clamp(rest[passive], min=1.0).unsqueeze(0)
        # elas already set to er for all struts

    fixing_enum = _fixing_to_enum(fixing)
    fix_t = torch.tensor(fixing_enum, dtype=torch.int8, device=dev)
    is_free = fix_t == _FIXING_FREE      # (N_atoms,)
    is_sym = fix_t == _FIXING_SYM

    sym_axis = symmetry_axes[0] if symmetry_axes else 0
    free_axes_mask = torch.ones(3, dtype=torch.float32, device=dev)
    free_axes_mask[sym_axis] = 0.0  # scale factor 0 on symmetry axis for sym atoms

    a_exp = a_idx.unsqueeze(0).unsqueeze(-1).expand(B, -1, 3)
    b_exp = b_idx.unsqueeze(0).unsqueeze(-1).expand(B, -1, 3)

    eps = 1e-8

    for _ in range(max_iter):
        pos_old = pos.clone()

        # (B, N_struts, 3)
        vect = pos[:, a_idx, :] - pos[:, b_idx, :]
        d1 = vect.norm(dim=2)                              # (B, N_struts)
        safe_d1 = d1.clamp(min=eps)
        b = (d1 - targ) / safe_d1                          # (B, N_struts)
        active = b > 0
        b_half = torch.where(active, b / 2.0, torch.zeros_like(b))

        f_vec = vect * (b_half * elas).unsqueeze(-1)       # (B, N_struts, 3)

        force = torch.zeros(B, n_atoms, 3, dtype=torch.float32, device=dev)
        force.scatter_add_(1, a_exp, -f_vec)
        force.scatter_add_(1, b_exp, +f_vec)

        # Per-atom active-elasticity sum: (B, N_atoms)
        wgt = torch.zeros(B, n_atoms, dtype=torch.float32, device=dev)
        active_elas = torch.where(active, elas, torch.zeros_like(elas))
        wgt.scatter_add_(1, a_idx.unsqueeze(0).expand(B, -1), active_elas)
        wgt.scatter_add_(1, b_idx.unsqueeze(0).expand(B, -1), active_elas)
        wgt_safe = wgt.clamp(min=eps).unsqueeze(-1)

        update = force / wgt_safe
        update = torch.where(
            wgt.unsqueeze(-1) > 0, update, torch.zeros_like(update)
        )

        # Apply by fixing kind: build a (N_atoms, 3) per-axis scale factor
        axis_scale = torch.zeros(n_atoms, 3, dtype=torch.float32, device=dev)
        axis_scale[is_free] = 1.0
        if is_sym.any():
            axis_scale[is_sym] = free_axes_mask  # broadcast (3,) → per-sym-atom

        pos = pos + update * axis_scale.unsqueeze(0)

        delta = (pos - pos_old).norm(dim=(1, 2))
        if delta.max() < tol:
            break

    return pos.cpu().numpy()
