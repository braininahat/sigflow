"""High-level TongueModel wrapper for loading, solving, and querying."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from sigflow.biomech.s3d_parser import S3DModel, parse_s3d
from sigflow.biomech.solver import solve_equilibrium


class TongueModel:
    """Biomechanical tongue model loaded from .s3d format."""

    def __init__(self, s3d_path: str | Path) -> None:
        self._model = parse_s3d(s3d_path)
        self._rest_positions = self._model.positions.copy()
        self._current_positions = self._model.positions.copy()
        self._activations = np.full(self._model.n_muscles, 100.0)

    @property
    def model(self) -> S3DModel:
        return self._model

    @property
    def positions(self) -> np.ndarray:
        return self._current_positions

    @property
    def rest_positions(self) -> np.ndarray:
        return self._rest_positions

    @property
    def activations(self) -> np.ndarray:
        return self._activations

    @property
    def muscle_names(self) -> list[str]:
        return self._model.muscle_names

    def set_activations(self, pcts: dict[str, float] | np.ndarray) -> None:
        """Set muscle activations. Dict keys are muscle names, values are percentages [0-100]."""
        if isinstance(pcts, dict):
            for name, pct in pcts.items():
                try:
                    idx = self._model.muscle_names.index(name)
                    self._activations[idx] = pct
                except ValueError:
                    pass
        else:
            self._activations[:] = pcts

    def solve(self, max_iter: int = 100) -> np.ndarray:
        """Run equilibrium solver with current activations. Returns updated positions."""
        self._current_positions = solve_equilibrium(
            positions=self._rest_positions,
            strut_pairs=self._model.strut_pairs,
            rest_lengths=self._model.rest_lengths,
            elasticity_r=self._model.elasticity_r,
            elasticity_c=self._model.elasticity_c,
            strut_muscles=self._model.strut_muscles,
            fixing=self._model.fixing,
            muscle_pcts=self._activations,
            max_iter=max_iter,
            symmetry_axes=self._model.symmetry_axes,
            symmetry_coord=self._model.symmetry_coord,
        )
        return self._current_positions

    def triangles(self) -> np.ndarray:
        """Get triangle index buffer (T, 3) int32."""
        return self._model.triangulate()

    def normals(self, positions: np.ndarray | None = None) -> np.ndarray:
        """Compute per-vertex normals for current or given positions."""
        return self._model.compute_normals(positions or self._current_positions)

    def midsagittal_curve(self, positions: np.ndarray | None = None) -> np.ndarray:
        """Get midsagittal surface positions (K, 3), sorted anterior-to-posterior."""
        if positions is None:
            positions = self._current_positions
        indices = self._model.midsagittal_surface()
        return positions[indices]

    def to_vertex_buffer(self, positions: np.ndarray | None = None) -> bytes:
        """Pack positions + normals into interleaved 24-byte stride buffer for Qt Quick 3D."""
        if positions is None:
            positions = self._current_positions
        norms = self._model.compute_normals(positions)
        # Interleave: [x, y, z, nx, ny, nz] per vertex, float32
        buf = np.empty((len(positions), 6), dtype=np.float32)
        buf[:, :3] = positions.astype(np.float32)
        buf[:, 3:] = norms.astype(np.float32)
        return buf.tobytes()

    def to_index_buffer(self) -> bytes:
        """Pack triangle indices into uint32 buffer for Qt Quick 3D."""
        tris = self.triangles()
        return tris.astype(np.uint32).tobytes()
