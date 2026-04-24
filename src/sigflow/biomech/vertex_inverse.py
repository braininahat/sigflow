"""Keypoint-displacement → full-mesh-vertex ridge.

A counterpart to `inverse.build_inverse_mapping`, which learns
`kp_disp → muscle_activations`.  This module instead learns
`kp_disp → vertex_displacement_for_every_mesh_atom`, skipping the
muscle intermediate entirely.

The motivation is runtime rendering: the forward solver is ~100 ms per
frame; a linear map from 22-dim displacement features to the full
mirrored vertex buffer is sub-µs, which leaves the 30 Hz budget intact.
The ridge is trained on the same `(activation → solved positions)`
synthetic pairs used by `build_inverse_mapping`, so both mappings are
consistent with the MyoSim3D forward solver.

Outputs are half-model positions mirrored across the symmetry axis
(`mesh.mirror_mesh`) so the Qt3D geometry can render the full tongue.
Triangulation (`mesh.triangulate_polygons`) is stored with the mapping
so a consumer can set index buffers once and stream positions each
frame without re-parsing the `.s3d`.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from .forward_solver import solve_equilibrium, solve_equilibrium_batch
from .inverse import resample_curve
from .mesh import atom_id_to_idx, mirror_mesh, triangulate_polygons
from .types import MyoSim3D


@dataclass
class VertexMapping:
    """Linear map from keypoint displacement to full mesh-vertex displacement.

    Inference: ``rest_positions + (W @ kp_disp + b).reshape(-1, 3)``.

    Attributes:
        W: weight matrix ``(3 · N_mirrored, 22)`` float32
        b: bias vector ``(3 · N_mirrored,)`` float32
        rest_positions: rest vertex positions ``(N_mirrored, 3)`` float32
        rest_keypoints: 11-point midsagittal rest contour ``(11, 2)`` float32
        tris: triangle indices into `rest_positions` ``(M, 3)`` uint32
        frozen_axis: symmetry axis index (0=X, 1=Y, 2=Z)
        midline_indices: atom indices of the half-model midsagittal contour
        alpha: ridge regularization coefficient
    """

    W: np.ndarray
    b: np.ndarray
    rest_positions: np.ndarray
    rest_keypoints: np.ndarray
    tris: np.ndarray
    frozen_axis: int
    midline_indices: np.ndarray
    alpha: float


def build_vertex_mapping(
    model: MyoSim3D,
    s3d_path: str | Path,
    *,
    n_samples: int = 5000,
    alpha: float = 1.0,
    use_torch: bool = True,
    batch_size: int = 256,
    verbose: bool = False,
    seed: Optional[int] = None,
) -> VertexMapping:
    """Build ridge-regression keypoint→vertex mapping.

    Mirrors `build_inverse_mapping` but targets the full mirrored mesh vertex
    buffer rather than muscle activations.  Triangulation is computed from the
    `.s3d` polygon records; the path is required because polygons reference
    atom IDs that must be mapped back to dense indices.

    Args:
        model: MyoSim3D tongue model
        s3d_path: path to the `.s3d` the model was parsed from (for polygons)
        n_samples: number of synthetic training samples
        alpha: ridge regularization (L2) coefficient
        use_torch: if True, use the GPU-batched forward solver; else CPU loop
        batch_size: batch size for the GPU forward solver
        verbose: print progress
        seed: random seed for reproducibility
    """
    rng = np.random.default_rng(seed)
    if seed is not None:
        np.random.seed(seed)

    frozen_axis = 0
    axes = [i for i in range(3) if i != frozen_axis]

    rest_half = model.positions.copy().astype(np.float32)
    rest_keypoints_2d = rest_half[model.midline_indices][:, axes]
    rest_keypoints_11 = resample_curve(rest_keypoints_2d, n_out=11).astype(np.float32)

    aid_to_idx = atom_id_to_idx(str(s3d_path))
    tris_half = triangulate_polygons(model.polygons or {}, aid_to_idx)
    if len(tris_half) == 0:
        raise ValueError(f"no triangles extracted from polygons in {s3d_path}")

    rest_full, tris_full = mirror_mesh(rest_half, tris_half, axis=frozen_axis)
    n_full = rest_full.shape[0]

    if verbose:
        print(f"  mesh: {n_full} mirrored vertices, {len(tris_full)} triangles")

    # === Generate training positions via forward solver ===
    activations_train = rng.uniform(40, 100, size=(n_samples, 23)).astype(np.float32)

    if verbose:
        print(f"  forward-solving {n_samples} samples...")

    positions_train: np.ndarray
    if use_torch:
        try:
            n_batches = (n_samples + batch_size - 1) // batch_size
            chunks = []
            for batch_idx in range(n_batches):
                start = batch_idx * batch_size
                end = min(start + batch_size, n_samples)
                chunk = solve_equilibrium_batch(
                    positions=model.positions,
                    strut_pairs=model.strut_pairs,
                    rest_lengths=model.rest_lengths,
                    elasticity_r=model.elasticity_r,
                    elasticity_c=model.elasticity_c,
                    strut_muscles=model.strut_muscles,
                    fixing=model.fixing_enum if model.fixing_enum is not None else model.fixing,
                    muscle_pcts_batch=activations_train[start:end],
                    max_iter=100,
                    symmetry_axes=model.symmetry_axes,
                    symmetry_coord=model.symmetry_coord,
                )
                chunks.append(chunk)
                if verbose:
                    print(f"    batch {batch_idx + 1}/{n_batches}")
            positions_train = np.concatenate(chunks, axis=0)
        except Exception:
            use_torch = False

    if not use_torch:
        positions_list = []
        for i, acts in enumerate(activations_train):
            pos = solve_equilibrium(
                positions=model.positions,
                strut_pairs=model.strut_pairs,
                rest_lengths=model.rest_lengths,
                elasticity_r=model.elasticity_r,
                elasticity_c=model.elasticity_c,
                strut_muscles=model.strut_muscles,
                fixing=model.fixing,
                muscle_pcts=acts,
                max_iter=100,
                symmetry_axes=model.symmetry_axes,
                symmetry_coord=model.symmetry_coord,
            )
            positions_list.append(pos)
            if verbose and (i + 1) % 100 == 0:
                print(f"    {i + 1}/{n_samples}")
        positions_train = np.array(positions_list)

    # positions_train: (n_samples, N_half, 3)

    # === Mirror each sample + build design matrices ===
    if verbose:
        print("  building design matrices...")

    X = np.zeros((n_samples, 22), dtype=np.float32)
    Y = np.zeros((n_samples, n_full * 3), dtype=np.float32)
    sign = np.ones(3, dtype=np.float32)
    sign[frozen_axis] = -1.0

    for i, half_positions in enumerate(positions_train):
        kp_2d = half_positions[model.midline_indices][:, axes]
        kp_11 = resample_curve(kp_2d, n_out=11)
        X[i] = (kp_11 - rest_keypoints_11).flatten()

        mirrored_half = half_positions * sign
        full_positions = np.concatenate([half_positions, mirrored_half], axis=0)
        Y[i] = (full_positions - rest_full).astype(np.float32).ravel()

    # === Ridge: W = (XᵀX + αI)⁻¹ Xᵀ Y ===
    if verbose:
        print(f"  solving ridge (alpha={alpha}, target dim={Y.shape[1]})...")

    XtX = X.T @ X
    XtY = X.T @ Y
    reg_mat = XtX + alpha * np.eye(22, dtype=np.float32)
    W_T = np.linalg.solve(reg_mat, XtY)          # (22, 3·N_full)
    W = W_T.T.astype(np.float32)                 # (3·N_full, 22)
    b = np.mean(Y, axis=0).astype(np.float32)    # (3·N_full,)

    return VertexMapping(
        W=W,
        b=b,
        rest_positions=rest_full.astype(np.float32),
        rest_keypoints=rest_keypoints_11,
        tris=tris_full.astype(np.uint32),
        frozen_axis=frozen_axis,
        midline_indices=np.asarray(model.midline_indices, dtype=np.int32),
        alpha=float(alpha),
    )


def predict_vertices(kp_disp: np.ndarray, mapping: VertexMapping) -> np.ndarray:
    """Predict full mirrored vertex positions from keypoint displacement.

    Args:
        kp_disp: flattened 11-point displacement ``(22,)``
        mapping: `VertexMapping` from `build_vertex_mapping` or `load_vertex_mapping`

    Returns:
        Vertex positions ``(N_mirrored, 3)`` float32.
    """
    kp_disp = np.asarray(kp_disp, dtype=np.float32)
    disp = mapping.W @ kp_disp + mapping.b
    return mapping.rest_positions + disp.reshape(-1, 3)


def save_vertex_mapping(mapping: VertexMapping, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        W=mapping.W,
        b=mapping.b,
        rest_positions=mapping.rest_positions,
        rest_keypoints=mapping.rest_keypoints,
        tris=mapping.tris,
        frozen_axis=np.array(mapping.frozen_axis),
        midline_indices=mapping.midline_indices,
        alpha=np.array(mapping.alpha),
    )


def load_vertex_mapping(path: str) -> VertexMapping:
    data = np.load(path)
    return VertexMapping(
        W=data["W"].astype(np.float32),
        b=data["b"].astype(np.float32),
        rest_positions=data["rest_positions"].astype(np.float32),
        rest_keypoints=data["rest_keypoints"].astype(np.float32),
        tris=data["tris"].astype(np.uint32),
        frozen_axis=int(data["frozen_axis"]),
        midline_indices=data["midline_indices"].astype(np.int32),
        alpha=float(data["alpha"]),
    )
