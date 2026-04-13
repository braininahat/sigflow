"""Parse MyoSim3D .s3d model files into NumPy arrays.

File format (text, line-oriented):
  P<id>: name, x, y, z, mass [, fixing [, rigid]]
  S<id>: name, a1, a2, rest_len, elast_r, elast_c, axis, muscle, color, pen
  M<id>: chart, top, label
  G<id>: n_verts, v1, v2, v3 [, v4] [, visible_side] [, color]
  H<id>: n_pos_faces, f1..., n_neg_faces, f1...
  R: [axis=]i, j: n, height [, active]
  Y: symmetry_axes, symmetry_coord
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass(kw_only=True)
class S3DModel:
    """Parsed .s3d model data in NumPy arrays."""

    # Atoms: positions and properties
    positions: np.ndarray  # (N, 3) float64
    masses: np.ndarray  # (N,) float64
    fixing: np.ndarray  # (N,) int32 — 0=free, 1=static, 2=symmetric
    atom_names: list[str] = field(default_factory=list)

    # Struts: connectivity and elasticity
    strut_pairs: np.ndarray  # (M, 2) int32 — atom index pairs (0-based)
    rest_lengths: np.ndarray  # (M,) float64
    elasticity_r: np.ndarray  # (M,) float64 — relaxed stiffness
    elasticity_c: np.ndarray  # (M,) float64 — contracted stiffness
    strut_muscles: np.ndarray  # (M,) int32 — muscle index (-1 = passive)

    # Muscles: names and count
    muscle_names: list[str] = field(default_factory=list)

    # Polygons: face connectivity (list of arrays, each [v1, v2, v3] or [v1, v2, v3, v4])
    polygons: list[np.ndarray] = field(default_factory=list)

    # Polyhedra: volume elements
    polyhedra: list[tuple[np.ndarray, np.ndarray]] = field(
        default_factory=list
    )  # (pos_faces, neg_faces)

    # Symmetry
    symmetry_axes: int = 0  # 0=XY, 1=XZ, 2=YZ
    symmetry_coord: float = 0.0

    @property
    def n_atoms(self) -> int:
        return len(self.positions)

    @property
    def n_struts(self) -> int:
        return len(self.strut_pairs)

    @property
    def n_muscles(self) -> int:
        return len(self.muscle_names)

    @property
    def n_polygons(self) -> int:
        return len(self.polygons)

    def triangulate(self) -> np.ndarray:
        """Convert polygon faces to triangle indices (0-based). Returns (T, 3) int32."""
        tris = []
        for face in self.polygons:
            if len(face) == 3:
                tris.append(face)
            elif len(face) >= 4:
                # Fan triangulation from first vertex
                for i in range(1, len(face) - 1):
                    tris.append([face[0], face[i], face[i + 1]])
        return np.array(tris, dtype=np.int32) if tris else np.empty((0, 3), dtype=np.int32)

    def compute_normals(self, positions: np.ndarray | None = None) -> np.ndarray:
        """Compute per-vertex normals from polygon faces. Returns (N, 3) float64."""
        if positions is None:
            positions = self.positions
        normals = np.zeros_like(positions)
        for face in self.polygons:
            if len(face) < 3:
                continue
            # 0-based indices
            v0, v1, v2 = face[0], face[1], face[2]
            e1 = positions[v1] - positions[v0]
            e2 = positions[v2] - positions[v0]
            fn = np.cross(e1, e2)
            for vi in face:
                normals[vi] += fn
        # Normalize
        lengths = np.linalg.norm(normals, axis=1, keepdims=True)
        lengths = np.maximum(lengths, 1e-10)
        return normals / lengths

    def midsagittal_surface(self, y_threshold: float = 30.0) -> np.ndarray:
        """Return indices of atoms near the midsagittal plane (Y ≈ 0), sorted by X."""
        near_midline = np.abs(self.positions[:, 1]) < y_threshold
        free = self.fixing != 1  # Exclude static atoms
        mask = near_midline & free
        indices = np.where(mask)[0]
        # Sort by X (anterior-posterior)
        return indices[np.argsort(self.positions[indices, 0])]


def parse_s3d(path: str | Path) -> S3DModel:
    """Parse a .s3d file into an S3DModel."""
    path = Path(path)

    # First pass: collect raw records by type
    raw_atoms: dict[int, tuple] = {}  # id -> (name, x, y, z, mass, fixing)
    raw_struts: dict[int, tuple] = {}
    raw_muscles: dict[int, str] = {}
    raw_polygons: dict[int, list[int]] = {}
    raw_polyhedra: dict[int, tuple] = {}
    symmetry_axes = 0
    symmetry_coord = 0.0

    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("Ver"):
                continue

            tag = line[0]
            if tag == "P":
                m = re.match(
                    r"P(\d+):([^,]*),([^,]+),([^,]+),([^,]+),([^,]+)(?:,(\d+))?", line
                )
                if m:
                    aid = int(m.group(1))
                    name = m.group(2).strip()
                    x, y, z = float(m.group(3)), float(m.group(4)), float(m.group(5))
                    mass = float(m.group(6))
                    fixing = int(m.group(7)) if m.group(7) else 0
                    raw_atoms[aid] = (name, x, y, z, mass, fixing)

            elif tag == "S":
                m = re.match(
                    r"S(\d+):([^,]*),(\d+),(\d+),([^,]+),([^,]+),([^,]+),(\d+),(-?\d+)",
                    line,
                )
                if m:
                    sid = int(m.group(1))
                    a1, a2 = int(m.group(3)), int(m.group(4))
                    rest = float(m.group(5))
                    er, ec = float(m.group(6)), float(m.group(7))
                    musc = int(m.group(9))
                    raw_struts[sid] = (a1, a2, rest, er, ec, musc)

            elif tag == "M":
                m = re.match(r"M(\d+):\d+,\d+,(.*)", line)
                if m:
                    raw_muscles[int(m.group(1))] = m.group(2).strip()

            elif tag == "G":
                m = re.match(r"G(\d+):(\d+),(.*)", line)
                if m:
                    gid = int(m.group(1))
                    n_verts = int(m.group(2))
                    rest = m.group(3)
                    parts = rest.split(",")
                    verts = []
                    for p in parts[:n_verts]:
                        p = p.strip()
                        if p.lstrip("-").isdigit():
                            verts.append(int(p))
                    raw_polygons[gid] = verts

            elif tag == "H":
                m = re.match(r"H(\d+):(.*)", line)
                if m:
                    hid = int(m.group(1))
                    parts = [p.strip() for p in m.group(2).split(",")]
                    idx = 0
                    n_pos = int(parts[idx])
                    idx += 1
                    pos_faces = [int(parts[idx + i]) for i in range(n_pos)]
                    idx += n_pos
                    n_neg = int(parts[idx])
                    idx += 1
                    neg_faces = [int(parts[idx + i]) for i in range(n_neg)]
                    raw_polyhedra[hid] = (pos_faces, neg_faces)

            elif tag == "Y":
                m = re.match(r"Y:(\d+),([^,]+)", line)
                if m:
                    symmetry_axes = int(m.group(1))
                    symmetry_coord = float(m.group(2))

    # Build ID → 0-based index mapping for atoms
    atom_ids = sorted(raw_atoms.keys())
    id_to_idx = {aid: i for i, aid in enumerate(atom_ids)}
    n_atoms = len(atom_ids)

    positions = np.zeros((n_atoms, 3), dtype=np.float64)
    masses = np.zeros(n_atoms, dtype=np.float64)
    fixing = np.zeros(n_atoms, dtype=np.int32)
    atom_names = []

    for i, aid in enumerate(atom_ids):
        name, x, y, z, mass, fix = raw_atoms[aid]
        positions[i] = [x, y, z]
        masses[i] = mass
        fixing[i] = fix
        atom_names.append(name)

    # Build strut arrays
    strut_ids = sorted(raw_struts.keys())
    n_struts = len(strut_ids)

    strut_pairs = np.zeros((n_struts, 2), dtype=np.int32)
    rest_lengths = np.zeros(n_struts, dtype=np.float64)
    elasticity_r = np.zeros(n_struts, dtype=np.float64)
    elasticity_c = np.zeros(n_struts, dtype=np.float64)
    strut_muscles = np.full(n_struts, -1, dtype=np.int32)

    for i, sid in enumerate(strut_ids):
        a1, a2, rest, er, ec, musc = raw_struts[sid]
        strut_pairs[i] = [id_to_idx.get(a1, 0), id_to_idx.get(a2, 0)]
        rest_lengths[i] = rest
        elasticity_r[i] = er
        elasticity_c[i] = ec
        strut_muscles[i] = musc

    # Build muscle list
    muscle_ids = sorted(raw_muscles.keys())
    muscle_names = [raw_muscles[mid] for mid in muscle_ids]

    # Build polygon list (convert atom IDs to 0-based indices)
    polygon_ids = sorted(raw_polygons.keys())
    polygons = []
    for gid in polygon_ids:
        verts = raw_polygons[gid]
        face = np.array([id_to_idx.get(v, 0) for v in verts if v > 0], dtype=np.int32)
        if len(face) >= 3:
            polygons.append(face)

    # Build polyhedra list
    polyhedra = []
    for hid in sorted(raw_polyhedra.keys()):
        pos_faces, neg_faces = raw_polyhedra[hid]
        polyhedra.append(
            (np.array(pos_faces, dtype=np.int32), np.array(neg_faces, dtype=np.int32))
        )

    return S3DModel(
        positions=positions,
        masses=masses,
        fixing=fixing,
        atom_names=atom_names,
        strut_pairs=strut_pairs,
        rest_lengths=rest_lengths,
        elasticity_r=elasticity_r,
        elasticity_c=elasticity_c,
        strut_muscles=strut_muscles,
        muscle_names=muscle_names,
        polygons=polygons,
        polyhedra=polyhedra,
        symmetry_axes=symmetry_axes,
        symmetry_coord=symmetry_coord,
    )
