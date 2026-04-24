"""Mesh helpers for MyoSim3D `.s3d` → renderable triangle mesh.

The `.s3d` format specifies atoms (nodes), polygons (vertex loops), and a
symmetry axis.  The MyoSim3D solver operates on atoms; to render the tongue
we need triangulated vertex buffers with per-vertex normals, mirrored across
the symmetry axis to produce a full (non-half) mesh.

These helpers are the minimal shared vocabulary between:
- offline demos that visualize `.s3d` models (ultraspeech-pyqt, ultrasuite-analysis)
- the runtime `biomech_s3d_tongue_display` sink node
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np


def atom_id_to_idx(s3d_path: str | Path) -> dict[int, int]:
    """Read atom IDs (P-lines) from an `.s3d` and map each ID to its dense index."""
    ids: list[int] = []
    with open(s3d_path) as f:
        for line in f:
            if line.startswith("P"):
                m = re.match(r"P(\d+):", line)
                if m:
                    ids.append(int(m.group(1)))
    return {aid: i for i, aid in enumerate(sorted(set(ids)))}


def triangulate_polygons(
    polygons: dict, aid_to_idx: dict[int, int]
) -> np.ndarray:
    """Fan-triangulate each polygon (vertex-loop) into `(N, 3)` uint32 indices.

    Polygons whose atom IDs aren't in `aid_to_idx` are silently skipped.
    """
    tris: list[list[int]] = []
    for poly in polygons.values():
        if len(poly) < 3:
            continue
        try:
            idxs = [aid_to_idx[aid] for aid in poly]
        except KeyError:
            continue
        for j in range(1, len(idxs) - 1):
            tris.append([idxs[0], idxs[j], idxs[j + 1]])
    return np.array(tris, dtype=np.uint32)


def mirror_mesh(
    positions: np.ndarray, tris: np.ndarray, axis: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Mirror a half-mesh across `axis` (default X) and stitch triangles.

    The mirrored triangle winding is reversed (`[0, 2, 1]`) so face normals
    point outward on both halves.
    """
    n = positions.shape[0]
    sign = np.ones(3, dtype=positions.dtype)
    sign[axis] = -1.0
    mirrored = positions * sign
    positions_full = np.concatenate([positions, mirrored], axis=0)

    if len(tris) > 0:
        flipped = tris[:, [0, 2, 1]] + n
        tris_full = np.concatenate([tris, flipped], axis=0).astype(tris.dtype)
    else:
        tris_full = tris
    return positions_full, tris_full


def compute_vertex_normals(positions: np.ndarray, tris: np.ndarray) -> np.ndarray:
    """Area-weighted per-vertex normals.  Returns float32 `(N, 3)`."""
    normals = np.zeros_like(positions, dtype=np.float32)
    v0 = positions[tris[:, 0]]
    v1 = positions[tris[:, 1]]
    v2 = positions[tris[:, 2]]
    face_n = np.cross(v1 - v0, v2 - v0)
    np.add.at(normals, tris[:, 0], face_n)
    np.add.at(normals, tris[:, 1], face_n)
    np.add.at(normals, tris[:, 2], face_n)
    lens = np.linalg.norm(normals, axis=1, keepdims=True)
    lens[lens < 1e-8] = 1.0
    return (normals / lens).astype(np.float32)
