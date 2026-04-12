"""Vectorized weighted Jacobi relaxation solver for biomechanical tongue models.

Ported from MyoSim3D CalcOneIntervalSolve (Unit1.pas:5899-6502).
"""

from __future__ import annotations

import numpy as np


def compute_activations(
    rest_lengths: np.ndarray,
    elasticity_r: np.ndarray,
    elasticity_c: np.ndarray,
    strut_muscles: np.ndarray,
    muscle_pcts: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute target lengths and elasticities from muscle activation percentages.

    Args:
        rest_lengths: (M,) rest lengths per strut
        elasticity_r: (M,) relaxed stiffness per strut
        elasticity_c: (M,) contracted stiffness per strut
        strut_muscles: (M,) muscle index per strut (-1 = passive)
        muscle_pcts: (K,) activation percentages [0-100] per muscle

    Returns:
        target_lengths: (M,) target lengths
        elasticity: (M,) interpolated elasticities
    """
    target_lengths = rest_lengths.copy()
    elasticity = elasticity_r.copy()

    has_muscle = strut_muscles >= 0
    musc_idx = strut_muscles[has_muscle]
    pcts = muscle_pcts[musc_idx]

    target_lengths[has_muscle] = np.maximum(rest_lengths[has_muscle] * pcts / 100.0, 1.0)
    ratio = target_lengths[has_muscle] / rest_lengths[has_muscle]
    elasticity[has_muscle] = (
        (elasticity_r[has_muscle] - elasticity_c[has_muscle]) * ratio
        + elasticity_c[has_muscle]
    )

    return target_lengths, elasticity


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
    symmetry_axes: int = 0,
    symmetry_coord: float = 0.0,
) -> np.ndarray:
    """Run weighted Jacobi relaxation to find equilibrium positions.

    Args:
        positions: (N, 3) initial atom positions
        strut_pairs: (M, 2) atom index pairs (0-based)
        rest_lengths: (M,) rest lengths
        elasticity_r: (M,) relaxed stiffness
        elasticity_c: (M,) contracted stiffness
        strut_muscles: (M,) muscle index per strut (-1 = passive)
        fixing: (N,) fixing mode (0=free, 1=static, 2=symmetric)
        muscle_pcts: (K,) activation percentages [0-100]
        max_iter: number of relaxation iterations
        symmetry_axes: 0=XY, 1=XZ, 2=YZ
        symmetry_coord: coordinate of symmetry plane

    Returns:
        (N, 3) equilibrium positions
    """
    pos = positions.copy()
    n_atoms = len(pos)
    a1 = strut_pairs[:, 0]
    a2 = strut_pairs[:, 1]

    target_lengths, elasticity = compute_activations(
        rest_lengths, elasticity_r, elasticity_c, strut_muscles, muscle_pcts
    )

    free_mask = fixing == 0
    sym_mask = fixing == 2

    for _ in range(max_iter):
        # Compute displacement vectors
        diff = pos[a1] - pos[a2]  # (M, 3)
        dist = np.linalg.norm(diff, axis=1)  # (M,)
        dist = np.maximum(dist, 1e-10)

        # Fractional error (only pull when stretched beyond target)
        b = (dist - target_lengths) / dist / 2.0  # (M,)
        active = b > 0
        b_active = b[active]

        if not np.any(active):
            break

        # Compute force contributions
        force_vec = diff[active] * (b_active * elasticity[active])[:, None]  # (A, 3)

        # Accumulate forces and weights per atom
        forces = np.zeros((n_atoms, 3), dtype=np.float64)
        weights = np.zeros(n_atoms, dtype=np.float64)

        np.subtract.at(forces, a1[active], force_vec)
        np.add.at(forces, a2[active], force_vec)
        np.add.at(weights, a1[active], elasticity[active])
        np.add.at(weights, a2[active], elasticity[active])

        # Normalize forces by weight
        nonzero = weights > 0
        forces[nonzero] /= weights[nonzero, None]

        # Update free atoms
        pos[free_mask] += forces[free_mask]

        # Update symmetric atoms (constrained to symmetry plane)
        if np.any(sym_mask):
            sym_force = forces[sym_mask].copy()
            if symmetry_axes == 0:  # XY plane: freeze Z
                sym_force[:, 2] = 0
            elif symmetry_axes == 1:  # XZ plane: freeze Y
                sym_force[:, 1] = 0
            elif symmetry_axes == 2:  # YZ plane: freeze X
                sym_force[:, 0] = 0
            pos[sym_mask] += sym_force

    return pos
