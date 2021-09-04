from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
from OpenGL.arrays import vbo

import glfw
from binvox import *

dx = [1, 1, 0, 0, 1, 1, 0]
dy = [0, 1, 1, 0, 0, 1, 1]
dz = [0, 0, 0, 1, 1, 1, 1]

scr_width, scr_height = 800, 600
filepath = './map/NewWorld1.obj_512.binvox'
with open(filepath, 'rb') as f:
    voxel = read_as_sparse(f)

vertex_shader = """
#version 330
in vec3 position;
in vec3 color;
uniform mat4 transform;
out vec3 newColor;
void main()
{
    gl_Position = transform * vec4(position, 1.0f);
    newColor = color;
}
"""
fragment_shader = """
#version 330
in vec3 newColor;
out vec4 outColor;
void main()
{
    outColor = vec4(newColor, 1.0f);
}
"""

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

indices = np.array([
        0, 1, 2, 3,  # v0-v1-v2-v3 (front)
        4, 5, 1, 0,  # v4-v5-v1-v0 (top)
        3, 2, 6, 7,  # v3-v2-v6-v7 (bottom)
        5, 4, 7, 6,  # v5-v4-v7-v6 (back)
        1, 5, 6, 2,  # v1-v5-v6-v2 (right)
        4, 0, 3, 7  # v4-v0-v3-v7 (left)
    ], dtype=np.int)

cubes = np.array()
cube_indices = np.array()

for val in voxel:
    tmp_vertices = vertices.copy()
    tmp_vertices[0::3] += val[0]
    tmp_vertices[1::3] += val[1]
    tmp_vertices[2::3] += val[2]
    np.vstack(cubes, tmp_vertices)

    tmp_indices = indices.copy()
    tmp_indices += indices + len(cube_indices)
    np.vstack(cube_indices, tmp_indices)


def _main():
    if not glfw.init():
        raise Exception("GLFW can not be initialized.")

    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    window = glfw.create_window(scr_width, scr_height, "Hello World", None, None)

    if not window:
        glfw.terminate()
        raise Exception("Failed to create GLFW window.")
    glfw.make_context_current(window)
    glfw.set_framebuffer_size_callback(window, framebuffer_size_callback)

    vbo = glGenBuffers(1)


    while not glfw.window_should_close(window):
        processInput(window)
        glClearColor(0.2, 0.3, 0.3, 1.0)
        glClear(GL_COLOR_BUFFER_BIT)



        glfw.swap_buffers(window)
        glfw.poll_events()
    glfw.terminate()


def framebuffer_size_callback(window, width, height):
    GL_VIEWPORT(0, 0, width, height)

def processInput(window):
    if glfw.get_key(window, glfw.KEY_ESCAPE) == glfw.PRESS:
        glfw.set_window_should_close(window, True)


if __name__ == '__main__':
    _main()
