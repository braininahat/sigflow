"""Biomechanical tongue model — .s3d parser, equilibrium solver, and model wrapper."""

from sigflow.biomech.inverse import (
    InverseMapping,
    batched_forward_solve,
    build_inverse_mapping,
    fit_ridge,
    generate_synthetic_dataset,
    load_inverse_mapping,
    midline_indices,
    predict_activations,
    resample_curve,
    save_inverse_mapping,
)
from sigflow.biomech.model import TongueModel
from sigflow.biomech.s3d_parser import parse_s3d
from sigflow.biomech.solver import solve_equilibrium

__all__ = [
    "InverseMapping",
    "TongueModel",
    "batched_forward_solve",
    "build_inverse_mapping",
    "fit_ridge",
    "generate_synthetic_dataset",
    "load_inverse_mapping",
    "midline_indices",
    "parse_s3d",
    "predict_activations",
    "resample_curve",
    "save_inverse_mapping",
    "solve_equilibrium",
]
