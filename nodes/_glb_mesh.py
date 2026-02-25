"""GLB (glTF Binary) parser — extract skinned mesh + static mesh data.

Pure functions, no external dependencies beyond numpy and struct.
Parses skinned mesh (vertices, normals, indices, skin weights,
inverse bind matrices, bone hierarchy) and named static meshes (jaws).
"""
import json
import struct

import numpy as np

# glTF component type → (numpy dtype, byte size)
_COMPONENT_TYPES = {
    5120: (np.int8, 1),
    5121: (np.uint8, 1),
    5122: (np.int16, 2),
    5123: (np.uint16, 2),
    5125: (np.uint32, 4),
    5126: (np.float32, 4),
}

# glTF accessor type → element count
_TYPE_COUNTS = {
    "SCALAR": 1,
    "VEC2": 2,
    "VEC3": 3,
    "VEC4": 4,
    "MAT2": 4,
    "MAT3": 9,
    "MAT4": 16,
}


def _read_accessor(gltf: dict, bin_data: bytes, accessor_idx: int) -> np.ndarray:
    """Read a glTF accessor into a numpy array."""
    acc = gltf["accessors"][accessor_idx]
    bv = gltf["bufferViews"][acc["bufferView"]]
    dtype, comp_size = _COMPONENT_TYPES[acc["componentType"]]
    count = acc["count"]
    n_components = _TYPE_COUNTS[acc["type"]]

    offset = bv.get("byteOffset", 0) + acc.get("byteOffset", 0)
    stride = bv.get("byteStride", comp_size * n_components)

    if stride == comp_size * n_components:
        # Tightly packed — fast path
        length = count * n_components * comp_size
        arr = np.frombuffer(bin_data, dtype=dtype, count=count * n_components, offset=offset)
    else:
        # Strided — read element by element
        arr = np.empty(count * n_components, dtype=dtype)
        for i in range(count):
            start = offset + i * stride
            chunk = np.frombuffer(bin_data, dtype=dtype, count=n_components, offset=start)
            arr[i * n_components:(i + 1) * n_components] = chunk

    if n_components > 1:
        arr = arr.reshape(count, n_components)
    return arr


def _compute_node_world_transforms(nodes: list[dict], root_children: list[int]) -> np.ndarray:
    """Compute world transforms for all nodes by traversing the scene graph.

    Returns (N, 4, 4) float32 array of world-space transforms.
    """
    n = len(nodes)
    local = np.zeros((n, 4, 4), dtype=np.float32)

    for i, node in enumerate(nodes):
        if "matrix" in node:
            local[i] = np.array(node["matrix"], dtype=np.float32).reshape(4, 4).T  # column-major → row-major
        else:
            t = np.array(node.get("translation", [0, 0, 0]), dtype=np.float32)
            r = np.array(node.get("rotation", [0, 0, 0, 1]), dtype=np.float32)  # xyzw
            s = np.array(node.get("scale", [1, 1, 1]), dtype=np.float32)

            # Quaternion to rotation matrix
            x, y, z, w = r
            local[i] = np.array([
                [1 - 2*(y*y + z*z), 2*(x*y - w*z),     2*(x*z + w*y),     t[0]],
                [2*(x*y + w*z),     1 - 2*(x*x + z*z), 2*(y*z - w*x),     t[1]],
                [2*(x*z - w*y),     2*(y*z + w*x),     1 - 2*(x*x + y*y), t[2]],
                [0,                 0,                  0,                  1],
            ], dtype=np.float32)
            # Apply scale
            local[i, :3, 0] *= s[0]
            local[i, :3, 1] *= s[1]
            local[i, :3, 2] *= s[2]

    # BFS to compute world transforms
    world = np.eye(4, dtype=np.float32)[np.newaxis].repeat(n, axis=0)
    # Build parent map
    parent = np.full(n, -1, dtype=np.int32)
    for i, node in enumerate(nodes):
        for child in node.get("children", []):
            parent[child] = i

    # Topological order via BFS from roots
    visited = np.zeros(n, dtype=bool)
    queue = []
    for i in range(n):
        if parent[i] == -1:
            queue.append(i)
            visited[i] = True
            world[i] = local[i]

    head = 0
    while head < len(queue):
        idx = queue[head]
        head += 1
        for child in nodes[idx].get("children", []):
            if not visited[child]:
                world[child] = world[idx] @ local[child]
                visited[child] = True
                queue.append(child)

    return world


def _extract_static_mesh(gltf: dict, bin_data: bytes, mesh_idx: int) -> dict:
    """Extract vertices, normals, and indices from a non-skinned mesh.

    Returns dict with:
        vertices: (V, 3) float32
        normals:  (V, 3) float32
        indices:  (F,)   uint32
    """
    mesh = gltf["meshes"][mesh_idx]
    prim = mesh["primitives"][0]
    vertices = _read_accessor(gltf, bin_data, prim["attributes"]["POSITION"]).astype(np.float32)
    normals = _read_accessor(gltf, bin_data, prim["attributes"]["NORMAL"]).astype(np.float32)
    indices = _read_accessor(gltf, bin_data, prim["indices"]).astype(np.uint32).ravel()
    return {"vertices": vertices, "normals": normals, "indices": indices}


def parse_glb(path: str, static_mesh_names: tuple[str, ...] = ("upper_jaw", "lower_jaw")) -> dict:
    """Parse a GLB file and extract the first skinned mesh + skeleton + named static meshes.

    Returns dict with:
        vertices:       (V, 3) float32 — rest-pose positions
        normals:        (V, 3) float32 — rest-pose normals
        indices:        (F,)   uint32  — triangle index buffer (flat)
        joint_indices:  (V, 4) uint8   — per-vertex joint indices
        joint_weights:  (V, 4) float32 — per-vertex joint weights
        inv_bind_matrices: (J, 4, 4) float32 — inverse bind matrices
        bone_rest_world:   (J, 4, 4) float32 — bone world transforms at rest pose
        bone_parents:      (J,) int32 — parent index per joint (-1 for roots)
        joint_names:       list[str] — joint names in skin order
        joint_node_indices: (J,) int32 — glTF node index per joint
        num_joints:        int
        num_vertices:      int
        static_meshes:     dict[str, dict] — named static meshes (vertices, normals, indices)
    """
    with open(path, "rb") as f:
        data = f.read()

    # GLB header
    magic, version, total_length = struct.unpack_from("<III", data, 0)
    assert magic == 0x46546C67, f"Not a GLB file (magic={hex(magic)})"

    # JSON chunk
    json_len, json_type = struct.unpack_from("<II", data, 12)
    assert json_type == 0x4E4F534A, "Expected JSON chunk"
    gltf = json.loads(data[20:20 + json_len])

    # BIN chunk
    bin_offset = 20 + json_len
    bin_len, bin_type = struct.unpack_from("<II", data, bin_offset)
    assert bin_type == 0x004E4942, "Expected BIN chunk"
    bin_data = data[bin_offset + 8:bin_offset + 8 + bin_len]

    nodes = gltf["nodes"]

    # Find first skinned mesh
    skin_idx = None
    mesh_idx = None
    for node in nodes:
        if "skin" in node and "mesh" in node:
            skin_idx = node["skin"]
            mesh_idx = node["mesh"]
            break
    assert skin_idx is not None, "No skinned mesh found in GLB"

    skin = gltf["skins"][skin_idx]
    mesh = gltf["meshes"][mesh_idx]
    prim = mesh["primitives"][0]

    # Read mesh data
    vertices = _read_accessor(gltf, bin_data, prim["attributes"]["POSITION"]).astype(np.float32)
    normals = _read_accessor(gltf, bin_data, prim["attributes"]["NORMAL"]).astype(np.float32)
    raw_indices = _read_accessor(gltf, bin_data, prim["indices"]).astype(np.uint32).ravel()
    joint_indices = _read_accessor(gltf, bin_data, prim["attributes"]["JOINTS_0"]).astype(np.uint8)
    joint_weights = _read_accessor(gltf, bin_data, prim["attributes"]["WEIGHTS_0"]).astype(np.float32)

    # Read inverse bind matrices
    ibm = _read_accessor(gltf, bin_data, skin["inverseBindMatrices"]).astype(np.float32)
    num_joints = len(skin["joints"])
    # glTF stores matrices column-major; reshape and transpose
    ibm = ibm.reshape(num_joints, 4, 4).transpose(0, 2, 1)  # → row-major

    # Joint names and node indices
    joint_node_indices = np.array(skin["joints"], dtype=np.int32)
    joint_names = [nodes[ni].get("name", f"joint_{i}") for i, ni in enumerate(joint_node_indices)]

    # Compute world transforms for all nodes
    scene = gltf["scenes"][gltf.get("scene", 0)]
    world_transforms = _compute_node_world_transforms(nodes, scene.get("nodes", []))

    # Extract bone world transforms in skin joint order
    bone_rest_world = world_transforms[joint_node_indices]

    # Build parent map in joint-index space
    # First build node→joint mapping
    node_to_joint = {int(ni): ji for ji, ni in enumerate(joint_node_indices)}
    bone_parents = np.full(num_joints, -1, dtype=np.int32)
    for ji, ni in enumerate(joint_node_indices):
        # Walk up the node hierarchy to find parent joint
        current = ni
        while True:
            # Find parent node
            parent_node = -1
            for pi, pnode in enumerate(nodes):
                if current in pnode.get("children", []):
                    parent_node = pi
                    break
            if parent_node == -1:
                break
            if parent_node in node_to_joint:
                bone_parents[ji] = node_to_joint[parent_node]
                break
            current = parent_node

    # Extract named static (non-skinned) meshes by node name
    static_meshes = {}
    for node_idx, node in enumerate(nodes):
        name = node.get("name", "")
        if name in static_mesh_names and "mesh" in node and "skin" not in node:
            static_meshes[name] = _extract_static_mesh(gltf, bin_data, node["mesh"])

    return {
        "vertices": vertices,
        "normals": normals,
        "indices": raw_indices,
        "joint_indices": joint_indices,
        "joint_weights": joint_weights,
        "inv_bind_matrices": ibm,
        "bone_rest_world": bone_rest_world,
        "bone_parents": bone_parents,
        "joint_names": joint_names,
        "joint_node_indices": joint_node_indices,
        "num_joints": num_joints,
        "num_vertices": len(vertices),
        "static_meshes": static_meshes,
    }
