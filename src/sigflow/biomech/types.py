"""Data types for biomechanical tongue model."""

from dataclasses import dataclass
from typing import Optional

import numpy as np


# PCA component labels for the 3 principal axes of muscle activation space
PCA_LABELS = ("Height", "Curvature", "Protrusion")


@dataclass(frozen=True)
class MyoSim3D:
    """MyoSim3D tongue model parsed from .s3d file."""

    # Atom positions: (N_atoms, 3) float array in model units
    positions: np.ndarray

    # Strut pairs: (N_struts, 2) int array of atom indices
    strut_pairs: np.ndarray

    # Rest lengths: (N_struts,) float array
    rest_lengths: np.ndarray

    # Elasticity radial (circumferential): (N_struts,) float array
    elasticity_r: np.ndarray

    # Elasticity circumferential (longitudinal): (N_struts,) float array
    elasticity_c: np.ndarray

    # Muscle IDs per strut: (N_struts,) int array (-1 = passive/fixed)
    strut_muscles: np.ndarray

    # Fixed atoms: (N_atoms,) bool array — True for chStatic atoms only
    # (equivalent to `fixing_enum == 1`). chSym atoms are NOT marked here;
    # they're in `sym_atoms` below so callers who don't need symmetry
    # semantics keep the old single-flag behaviour.
    fixing: np.ndarray

    # Muscle names: list[str], length = max(strut_muscles) + 1
    muscle_names: list[str]

    # Tri-state fixing enum per MyoSim3D (Unit1.pas:159):
    #   0 = chFree, 1 = chStatic, 2 = chSym.
    # Populated by the parser; optional so old pickles still load.
    fixing_enum: Optional[np.ndarray] = None

    # chSym atoms (symmetry-plane atoms) as (N_atoms,) bool mask.
    # These move only in axes perpendicular to the symmetry axis.
    sym_atoms: Optional[np.ndarray] = None

    # Symmetry axes: tuple of axes that are symmetric (e.g., (0,) for XZ symmetry)
    symmetry_axes: tuple[int, ...] = (0,)

    # Symmetry coordinate: which axis to enforce symmetry along (e.g., 1 for Y)
    symmetry_coord: int = 1

    # Polygons for rendering: dict[int, list[int]]
    polygons: Optional[dict] = None

    # Midline atom indices (sorted): list[int]
    # Used to extract the midsagittal contour from the 3D model
    midline_indices: Optional[list[int]] = None


@dataclass(frozen=True)
class InverseMapping:
    """Ridge-regression inverse mapping: keypoint displacement → muscle activations."""

    # Weight matrix: (N_muscles, 22) where 22 = 11 keypoints * 2 axes
    W: np.ndarray

    # Bias vector: (N_muscles,)
    b: np.ndarray

    # Rest-pose keypoints: (11, 2) in model space
    rest_keypoints: np.ndarray

    # Frozen axis: 0=X, 1=Y, or 2=Z (the axis that is symmetric)
    frozen_axis: int

    # Muscle names: list[str]
    muscle_names: list[str]

    # Midline atom indices (sorted): list[int]
    # Used to extract the midsagittal contour from the 3D model
    midline_indices: list[int]

    # Ridge regression parameter (alpha)
    alpha: float = 1.0

    # PCA basis for dimensionality reduction (computed offline on synthetic data).
    # Top 3 principal components of the 23-dim muscle activation space.
    # Labels: Height (PC1), Curvature (PC2), Protrusion (PC3).
    pca_components: Optional[np.ndarray] = None       # (3, 23) float32
    pca_mean: Optional[np.ndarray] = None             # (23,) float32
    pca_explained_variance: Optional[np.ndarray] = None  # (3,) float32 — variance (not ratio)
