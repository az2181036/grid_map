import numpy as np

vertices = np.array([
        -0.5, 0.5, 0.5,
        0.5, 0.5, 0.5,
        0.5, -0.5, 0.5,
        -0.5, -0.5, 0.5,  # v0-v1-v2-v3

        -0.5, 0.5, -0.5,
        0.5, 0.5, -0.5,
        0.5, -0.5, -0.5,
        -0.5, -0.5, -0.5  # v4-v5-v6-v7
    ], dtype=np.float32)


tmp_vertices = vertices.copy()
tmp_vertices[0::3] += 0
tmp_vertices[1::3] += 1
tmp_vertices[2::3] += 2
print(tmp_vertices)