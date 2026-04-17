"""Parse MyoSim3D .s3d model files."""

import re

import numpy as np

from .types import MyoSim3D


def parse_s3d(path: str) -> MyoSim3D:
    """Parse a .s3d model file into a MyoSim3D dataclass.

    Args:
        path: Path to .s3d file.

    Returns:
        MyoSim3D object with positions, struts, muscles, etc.
    """
    atoms = {}  # id -> (x, y, z, mass, fixing)
    struts = {}  # id -> (a1, a2, rest_len, elast_r, elast_c, muscle)
    muscles = {}  # id -> name
    polygons = {}  # id -> [atom_ids]

    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            tag = line[0]

            if tag == "P":
                # P<id>:name,x,y,z,mass,fixing[,rigid]
                # fixing per MyoSim3D TFixing = (chFree, chStatic, chSym),
                # i.e., 0 = free, 1 = static, 2 = symmetry plane
                m = re.match(
                    r"P(\d+):([^,]*),([^,]+),([^,]+),([^,]+),([^,]+),(\d+)", line
                )
                if m:
                    aid = int(m.group(1))
                    x, y, z = float(m.group(3)), float(m.group(4)), float(m.group(5))
                    mass = float(m.group(6))
                    fixing = int(m.group(7))  # 0, 1, or 2
                    atoms[aid] = (x, y, z, mass, fixing)
                    continue
                # Fallback: fixing column missing
                m = re.match(
                    r"P(\d+):([^,]*),([^,]+),([^,]+),([^,]+),([^,]+)", line
                )
                if m:
                    aid = int(m.group(1))
                    x, y, z = float(m.group(3)), float(m.group(4)), float(m.group(5))
                    mass = float(m.group(6))
                    atoms[aid] = (x, y, z, mass, 0)

            elif tag == "S":
                # S<id>:name,atom1,atom2,restLen,elastR,elastC,axis,musc,color,pen
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
                    struts[sid] = (a1, a2, rest, er, ec, musc)

            elif tag == "M":
                # M<id>:chart,top,name
                m = re.match(r"M(\d+):\d+,\d+,(.*)", line)
                if m:
                    muscles[int(m.group(1))] = m.group(2).strip()

            elif tag == "G":
                # G<id>:nVerts,v1,v2,...
                m = re.match(r"G(\d+):(\d+),(.*)", line)
                if m:
                    gid = int(m.group(1))
                    rest = m.group(3)
                    verts = [int(v) for v in rest.split(",") if v.strip().isdigit()]
                    polygons[gid] = verts[: int(m.group(2))]

    # Convert to numpy arrays
    atom_ids = sorted(atoms.keys())
    atom_id_to_idx = {aid: i for i, aid in enumerate(atom_ids)}

    positions = np.array(
        [atoms[aid][:3] for aid in atom_ids], dtype=np.float32
    )  # (N, 3)
    fixing_enum = np.array(
        [atoms[aid][4] for aid in atom_ids], dtype=np.int8
    )  # (N,) values in {0,1,2}
    fixing = fixing_enum == 1  # backward-compat: True only for chStatic
    sym_atoms = fixing_enum == 2  # midline / symmetry-plane atoms

    strut_ids = sorted(struts.keys())

    strut_pairs = []
    rest_lengths = []
    elasticity_r = []
    elasticity_c = []
    strut_muscles = []

    for sid in strut_ids:
        a1, a2, rest, er, ec, musc = struts[sid]
        strut_pairs.append([atom_id_to_idx[a1], atom_id_to_idx[a2]])
        rest_lengths.append(rest)
        elasticity_r.append(er)
        elasticity_c.append(ec)
        strut_muscles.append(musc)

    strut_pairs = np.array(strut_pairs, dtype=np.int32)  # (N_struts, 2)
    rest_lengths = np.array(rest_lengths, dtype=np.float32)  # (N_struts,)
    elasticity_r = np.array(elasticity_r, dtype=np.float32)  # (N_struts,)
    elasticity_c = np.array(elasticity_c, dtype=np.float32)  # (N_struts,)
    strut_muscles = np.array(strut_muscles, dtype=np.int32)  # (N_struts,)

    # Build muscle names list
    muscle_names = [""] * (int(strut_muscles.max()) + 1)
    for mid, name in muscles.items():
        if 0 <= mid < len(muscle_names):
            muscle_names[mid] = name

    # Find midsagittal midline: atoms closest to the sagittal plane (X ≈ 0)
    # sorted by Z (posterior to anterior)
    midline_indices = _find_midline(positions)

    return MyoSim3D(
        positions=positions,
        strut_pairs=strut_pairs,
        rest_lengths=rest_lengths,
        elasticity_r=elasticity_r,
        elasticity_c=elasticity_c,
        strut_muscles=strut_muscles,
        fixing=fixing,
        muscle_names=muscle_names,
        fixing_enum=fixing_enum,
        sym_atoms=sym_atoms,
        symmetry_axes=(0,),
        symmetry_coord=1,
        polygons=polygons,
        midline_indices=midline_indices,
    )


def _find_midline(positions: np.ndarray, n_points: int = 25) -> list[int]:
    """Find midsagittal midline by selecting atoms closest to symmetry plane.

    For the tongue model with X-Y symmetry (X ≈ 0, Y varying, Z varying):
    - Find atoms with small |X| coordinate (close to midline)
    - Sort by Z (superior to inferior)
    - Return indices

    Args:
        positions: (N, 3) atom positions
        n_points: Target number of midline points

    Returns:
        List of atom indices representing midsagittal contour
    """
    # Find atoms near X ≈ 0 (within 10% of total X range)
    x_range = positions[:, 0].max() - positions[:, 0].min()
    threshold = 0.1 * x_range
    candidate_indices = np.where(np.abs(positions[:, 0]) < threshold)[0]

    if len(candidate_indices) < n_points:
        # Not enough atoms near midline; use all candidates
        return sorted(candidate_indices.tolist())

    # Among candidates, sort by Z (superior to inferior, i.e., descending Z)
    z_coords = positions[candidate_indices, 2]
    sorted_indices = candidate_indices[np.argsort(-z_coords)]

    # Resample to n_points by taking every k-th point
    step = len(sorted_indices) // n_points
    if step < 1:
        step = 1
    midline = sorted_indices[::step].tolist()

    return midline[:n_points]
