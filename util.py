import math
import numpy as np


def get_coordinate(t, mod1, mod2):
    x = math.floor(t / mod1)
    y = math.floor((t - x * mod1) / mod2)
    z = (t - x * mod1 - y * mod2)
    return x, y, z

def get_ABC(x, y):
    a = y[1] - y[0]
    b = x[0] - x[1]
    c = x[1] * y[0] - x[0] * y[1]
    return a, b, c


def dot(vector1, vector2):
    """
    :return: The dot (or scalar) product of the two vectors
    """
    return vector1[0] * vector2[0] + vector1[1] * vector2[1]


def orthogonal(vector):
    return vector[1], -vector[0]


def edge_direction(point0, point1):
    return [point1[0] - point0[0], point1[1] - point0[1], ]


def vertices_to_edges(vertices):
    return np.array(edge_direction(vertices[i], vertices[(i + 1) % len(vertices)])
            for i in range(len(vertices)))


def project(vertices, axis):
    """
    :return: A vector showing how much of the vertices lies along the axis
    """
    dots = [np.dot(vertex, axis) for vertex in vertices]
    return min(dots), max(dots)


def overlap(projection1, projection2):
    """
    :return: Boolean indicating if the two projections overlap
    """
    return min(projection1) <= max(projection2) and \
           min(projection2) <= max(projection1)


def separating_axis_theorem(vertices_a, vertices_b, normal):
    edges = np.append(vertices_to_edges(vertices_a) + vertices_to_edges(vertices_b))
    axes = [np.linalg.norm(orthogonal(edge)) for edge in edges]

    for axis in axes:
        projection_a = project(vertices_a, axis)
        projection_b = project(vertices_b, axis)

        overlapping = overlap(projection_a, projection_b)

        if not overlapping:
            return False

    return True


def main():
    a_vertices = [(0, 0, 70), (70, 0, 0), (0, 70,0)]
    b_vertices = [(70, 70), (150, 70), (70, 150)]
    c_vertices = [(30, 30, 0), (150, 70, 0), (70, 0, 150)]

    print(separating_axis_theorem(a_vertices, c_vertices))


if __name__ == "__main__":
    main()