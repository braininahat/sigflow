"""Biomechanical inverse mapping — keypoint displacement → muscle activations."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from .forward_solver import solve_equilibrium, solve_equilibrium_batch
from .types import InverseMapping, MyoSim3D


def build_inverse_mapping(
    model: MyoSim3D,
    n_samples: int = 5000,
    alpha: float = 1.0,
    use_torch: bool = True,
    batch_size: int = 256,
    verbose: bool = False,
    seed: Optional[int] = None,
) -> InverseMapping:
    """Build ridge-regression inverse mapping from synthetic data.

    Generates random muscle activation samples, runs the forward solver to
    get resulting tongue shapes, extracts 11-point midsagittal keypoints,
    and trains a ridge regression model on displacement features.

    Args:
        model: MyoSim3D tongue model
        n_samples: Number of synthetic training samples
        alpha: Ridge regression parameter (L2 penalty)
        use_torch: If True, use GPU-batched forward solver; else CPU
        batch_size: Batch size for GPU forward solver
        verbose: Print progress
        seed: Random seed for reproducibility

    Returns:
        InverseMapping with trained W, b matrices
    """
    if seed is not None:
        np.random.seed(seed)

    rng = np.random.default_rng(seed)

    # === 1. Extract midsagittal contour structure ===
    frozen_axis = 0  # X-Y plane (sagittal)
    axes = [i for i in range(3) if i != frozen_axis]  # [1, 2]

    rest_positions = model.positions.copy()
    rest_keypoints_2d = rest_positions[model.midline_indices][:, axes]
    rest_keypoints_11 = resample_curve(rest_keypoints_2d, n_out=11)

    # === 2. Generate training data ===
    if verbose:
        print(f"Generating {n_samples} synthetic samples...")

    # Random activations: uniform in [40, 100] % (mid-range for tongue)
    activations_train = rng.uniform(40, 100, size=(n_samples, 23)).astype(np.float32)
    displacements_train = []

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
                    print(f"  Batch {batch_idx + 1}/{n_batches}")
            positions_train = np.concatenate(chunks, axis=0)  # (n_samples, N_atoms, 3)

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
                print(f"  {i + 1}/{n_samples}")

        positions_train = np.array(positions_list)  # (n_samples, N_atoms, 3)

    # === 3. Extract midsagittal keypoints and displacement features ===
    if verbose:
        print("Extracting keypoint features...")

    for pos in positions_train:
        kp_2d = pos[model.midline_indices][:, axes]
        kp_11 = resample_curve(kp_2d, n_out=11)
        disp = (kp_11 - rest_keypoints_11).flatten()
        displacements_train.append(disp)

    X = np.array(displacements_train, dtype=np.float32)  # (n_samples, 22)
    y = activations_train  # (n_samples, 23)

    # === 4. Ridge regression ===
    if verbose:
        print(f"Training ridge regression (alpha={alpha})...")

    # (X^T X + alpha*I)^{-1} X^T y
    XtX = X.T @ X  # (22, 22)
    Xty = X.T @ y  # (22, 23)
    reg_mat = XtX + alpha * np.eye(22, dtype=np.float32)
    W = np.linalg.solve(reg_mat, Xty).T  # (23, 22)
    b = np.mean(y, axis=0)  # (23,)

    if verbose:
        print(f"  W shape: {W.shape}, b shape: {b.shape}")

    return InverseMapping(
        W=W.astype(np.float32),
        b=b.astype(np.float32),
        rest_keypoints=rest_keypoints_11.astype(np.float32),
        frozen_axis=frozen_axis,
        muscle_names=model.muscle_names,
        midline_indices=model.midline_indices,
        alpha=alpha,
    )


def predict_activations(
    displacement_features: np.ndarray,
    W: np.ndarray,
    b: np.ndarray,
) -> np.ndarray:
    """Predict muscle activations from keypoint displacement.

    Fast inference: single matrix multiplication + clipping.

    Args:
        displacement_features: Flattened 11-point displacement (22,) float32
        W: Weight matrix (22, 23) float32
        b: Bias vector (23,) float32

    Returns:
        Predicted activations (23,) float32, clipped to [0, 100]
    """
    displacement_features = np.asarray(displacement_features, dtype=np.float32)
    activations = displacement_features @ W + b
    return np.clip(activations, 0, 100).astype(np.float32)


def save_inverse_mapping(mapping: InverseMapping, path: str) -> None:
    """Save inverse mapping to .npz file.

    Args:
        mapping: InverseMapping object
        path: Output file path
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    arrays: dict = dict(
        W=mapping.W,
        b=mapping.b,
        rest_keypoints=mapping.rest_keypoints,
        frozen_axis=np.array(mapping.frozen_axis),
        muscle_names=np.array(mapping.muscle_names, dtype=object),
        midline_indices=np.array(mapping.midline_indices, dtype=np.int32),
        alpha=np.array(mapping.alpha),
    )
    if mapping.pca_components is not None:
        arrays["pca_components"] = mapping.pca_components
    if mapping.pca_mean is not None:
        arrays["pca_mean"] = mapping.pca_mean
    if mapping.pca_explained_variance is not None:
        arrays["pca_explained_variance"] = mapping.pca_explained_variance
    np.savez(path, **arrays)


def load_inverse_mapping(path: str) -> InverseMapping:
    """Load inverse mapping from .npz file.

    Args:
        path: Input file path

    Returns:
        InverseMapping object
    """
    data = np.load(path, allow_pickle=True)
    return InverseMapping(
        W=data["W"].astype(np.float32),
        b=data["b"].astype(np.float32),
        rest_keypoints=data["rest_keypoints"].astype(np.float32),
        frozen_axis=int(data["frozen_axis"]),
        muscle_names=list(data["muscle_names"]),
        midline_indices=list(data["midline_indices"].astype(int)),
        alpha=float(data["alpha"]),
        pca_components=data["pca_components"].astype(np.float32) if "pca_components" in data else None,
        pca_mean=data["pca_mean"].astype(np.float32) if "pca_mean" in data else None,
        pca_explained_variance=data["pca_explained_variance"].astype(np.float32) if "pca_explained_variance" in data else None,
    )


def resample_curve(curve: np.ndarray, n_out: int = 11) -> np.ndarray:
    """Resample 2D curve to fixed number of points via linear interpolation.

    Args:
        curve: Input curve (N, 2) float
        n_out: Target number of points

    Returns:
        Resampled curve (n_out, 2) float
    """
    curve = np.asarray(curve, dtype=np.float32)
    n_in = len(curve)

    if n_in == n_out:
        return curve

    if n_in < 2:
        raise ValueError("Curve must have at least 2 points")

    # Parameterize by arc length
    diffs = np.diff(curve, axis=0)
    dists = np.linalg.norm(diffs, axis=1)
    cumsum = np.concatenate([[0], np.cumsum(dists)])
    total_len = cumsum[-1]

    if total_len < 1e-6:
        # Degenerate curve (all points coincide) — return copies of first point
        return np.tile(curve[0:1], (n_out, 1))

    # Interpolate
    t_old = cumsum / total_len  # 0 to 1
    t_new = np.linspace(0, 1, n_out)  # 0 to 1

    resampled = np.zeros((n_out, 2), dtype=np.float32)
    for dim in range(2):
        resampled[:, dim] = np.interp(t_new, t_old, curve[:, dim])

    return resampled
