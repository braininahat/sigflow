"""Biomechanical tongue model inference — inverse mapping + forward solver."""

from .forward_solver import (
    solve_equilibrium,
    solve_equilibrium_batch,
    solve_equilibrium_rollout,
)
from .gno_model import BiomechGINO, BiomechStepGNO
from .inverse import (
    build_inverse_mapping,
    load_inverse_mapping,
    predict_activations,
    resample_curve,
    save_inverse_mapping,
)
from .mesh import (
    atom_id_to_idx,
    compute_vertex_normals,
    mirror_mesh,
    triangulate_polygons,
)
from .s3d_parser import parse_s3d
from .types import InverseMapping, MyoSim3D, PCA_LABELS

__all__ = [
    "BiomechGINO",
    "BiomechStepGNO",
    "MyoSim3D",
    "InverseMapping",
    "PCA_LABELS",
    "atom_id_to_idx",
    "build_inverse_mapping",
    "compute_vertex_normals",
    "load_inverse_mapping",
    "mirror_mesh",
    "parse_s3d",
    "predict_activations",
    "resample_curve",
    "save_inverse_mapping",
    "solve_equilibrium",
    "solve_equilibrium_batch",
    "solve_equilibrium_rollout",
    "triangulate_polygons",
]
