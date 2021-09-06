import numpy as np

from binvox import *


def get_vertices_info(filepath):
    with open(filepath, 'rb') as f:
        voxel = read_as_sparse(f)

    vertices = np.array([
        -0.5, 0.5, 0.5, -1, 1, 1,
        0.5, 0.5, 0.5, 1, 1, 1,
        0.5, -0.5, 0.5, 1, -1, 1,
        -0.5, -0.5, 0.5, -1, -1, 1,
        -0.5, 0.5, -0.5, -1, 1, -1,
        0.5, 0.5, -0.5, 1, 1, -1,
        0.5, -0.5, -0.5, 1, -1, -1
        -0.5, -0.5, -0.5, -1, -1, -1
    ], dtype=np.float32)

    indices = np.array([
        0, 1, 2, 3,  # v0-v1-v2-v3 (front)
        4, 5, 1, 0,  # v4-v5-v1-v0 (top)
        3, 2, 6, 7,  # v3-v2-v6-v7 (bottom)
        5, 4, 7, 6,  # v5-v4-v7-v6 (back)
        1, 5, 6, 2,  # v1-v5-v6-v2 (right)
        4, 0, 3, 7  # v4-v0-v3-v7 (left)
    ], dtype=np.int)

    textures = np.array([
        0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0
    ] * len(voxel) * 6, dtype=np.float32)

    cubes = []
    cube_indices = []
    cnt = 0

    for val in voxel:
        tmp_vertices = vertices.copy()
        tmp_vertices[0::6] += val[0]
        tmp_vertices[1::6] += val[1]
        tmp_vertices[2::6] += val[2]
        cubes.extend(tmp_vertices.tolist())

        tmp_indices = indices.copy()
        tmp_indices += 8 * cnt
        cube_indices.extend(tmp_indices.tolist())
        cnt += 1

    cubes = np.array(cubes, dtype=np.float32)
    cube_indices = np.array(cube_indices, dtype=np.int)

    return cubes, cube_indices
