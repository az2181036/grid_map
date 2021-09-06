import ctypes

import OpenGL.GL.shaders
import numpy as np
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
from OpenGL.arrays import vbo

import glfw
import glm
import math

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
in layout(location=0) vec3 position;
uniform mat4 view;
uniform mat4 model;
uniform mat4 projection;
void main()
{
    gl_Position =  projection * view * model * vec4(position, 1.0f);
}
"""

fragment_shader = """
#version 330
out vec4 outColor;
void main()
{
    outColor = vec4(0.0, 1.0, 0.0, 1.0);
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

cubes = []
cube_indices = []
cnt = 0

for val in voxel:
    tmp_vertices = vertices.copy()
    tmp_vertices[0::3] += val[0]
    tmp_vertices[1::3] += val[1]
    tmp_vertices[2::3] += val[2]
    cubes.extend(tmp_vertices.tolist())

    tmp_indices = indices.copy()
    tmp_indices += 8 * cnt
    cube_indices.extend(tmp_indices.tolist())
    cnt += 1

cubes = np.array(cubes, dtype=np.float32)
cube_indices = np.array(cube_indices, dtype=np.int)

# cubes = np.array([
#     -0.5, 3.5, 250.5,
#     0.5, 3.5, 250.5,
#     0.5, 2.5, 250.5,
#     -0.5, 2.5, 250.5,
#
#     -0.5, 3.5, 249.5,
#     0.5, 3.5, 249.5,
#     0.5, 2.5, 249.5,
#     -0.5, 2.5, 249.5,
#
#     -0.5, 93.5, 249.5,
#     0.5, 93.5, 249.5,
#     0.5, 92.5, 249.5,
#     -0.5, 92.5, 249.5,
#
#     -0.5, 93.5, 248.5,
#     0.5, 93.5, 248.5,
#     0.5, 92.5, 248.5,
#     -0.5, 92.5, 248.5
# ], dtype=np.float32)
# cubes_indices = np.array([
#         0, 1, 2, 3,  # v0-v1-v2-v3 (front)
#         4, 5, 1, 0,  # v4-v5-v1-v0 (top)
#         3, 2, 6, 7,  # v3-v2-v6-v7 (bottom)
#         5, 4, 7, 6,  # v5-v4-v7-v6 (back)
#         1, 5, 6, 2,  # v1-v5-v6-v2 (right)
#         4, 0, 3, 7,  # v4-v0-v3-v7 (left)
# ], dtype=np.int)
# cube_indices = np.hstack((cubes_indices, cubes_indices+8))


cameraPos = glm.vec3(256, 0, 512)
cameraFront = glm.vec3(0, 0, -1)
cameraUp = glm.vec3(0, -1, 0)
deltaTime = 0
lastTime = 0
fov = 45.0
yaw, pitch = 0, 0


def main():
    global deltaTime, lastTime, cameraPos, cameraFront

    if not glfw.init():
        raise Exception("GLFW can not be initialized.")

    # glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    # glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    # glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    window = glfw.create_window(scr_width, scr_height, "Hello World", None, None)

    if not window:
        glfw.terminate()
        raise Exception("Failed to create GLFW window.")
    glfw.make_context_current(window)
    glfw.set_framebuffer_size_callback(window, framebuffer_size_callback)
    glfw.set_input_mode(window, glfw.CURSOR, glfw.CURSOR_DISABLED)
    # glfw.set_cursor_pos_callback(window, mouse_callback)
    glfw.set_scroll_callback(window, scroll_callback)
    shader = OpenGL.GL.shaders.compileProgram(OpenGL.GL.shaders.compileShader(vertex_shader, GL_VERTEX_SHADER),
                                              OpenGL.GL.shaders.compileShader(fragment_shader, GL_FRAGMENT_SHADER))

    vbo = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    glBufferData(GL_ARRAY_BUFFER, cubes.itemsize*len(cubes), cubes, GL_STATIC_DRAW)

    ebo = glGenBuffers(1)
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo)
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, cube_indices.itemsize * len(cube_indices), cube_indices, GL_STATIC_DRAW)

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 12, ctypes.c_void_p(0))
    glEnableVertexAttribArray(0)

    glUseProgram(shader)
    glEnable(GL_DEPTH_TEST)
    glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
    # glPolygonOffset(-1.0, -1.0)

    model = glm.mat4(1.0)
    projection = glm.perspective(glm.radians(fov), float(scr_width) / float(scr_height), 0.1, 512)

    view_loc = glGetUniformLocation(shader, "view")
    proj_loc = glGetUniformLocation(shader, "projection")
    model_loc = glGetUniformLocation(shader, "model")

    glUniformMatrix4fv(model_loc, 1, GL_FALSE, glm.value_ptr(model))
    glUniformMatrix4fv(proj_loc, 1, GL_FALSE, glm.value_ptr(projection))


    while not glfw.window_should_close(window):
        currentTime = glfw.get_time()
        deltaTime = currentTime - lastTime
        lastTime = currentTime

        glClearColor(1.0, 1.0, 1.0, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glViewport(0, 0, scr_width, scr_height)

        processInput(window)

        print(cameraPos)
        view = glm.lookAt(cameraPos,
                          glm.vec3(256, 256, 0),
                          cameraUp)
        glUniformMatrix4fv(view_loc, 1, GL_FALSE, glm.value_ptr(view))

        glDrawElements(GL_QUADS, len(cube_indices), GL_UNSIGNED_INT, None)
        # glDrawElements(GL_LINE_LOOP, len(cube_indices), GL_UNSIGNED_INT, None)
        # glDrawElements(GL_LINES, len(cube_indices), GL_UNSIGNED_INT, None)
        glfw.swap_buffers(window)
        glfw.poll_events()
    glfw.terminate()


firstMouse = True
lastX, lastY = 0, 0
def mouse_callback(window, xpos, ypos):
    global firstMouse, lastX, lastY, yaw, pitch, cameraFront
    if firstMouse:
        lastX, lastY = xpos, ypos
        firstMouse = False
        return
    xoffset, yoffset = xpos - lastX, ypos - lastY
    lastX, lastY = xpos, ypos

    xoffset *= 0.05
    yoffset *= 0.05

    yaw += xoffset
    pitch += yoffset
    pitch = pitch if pitch > -89.0 else -89.0
    pitch = pitch if pitch < 89.0 else 89.0

    front = glm.vec3(0, 0, 0)
    front.x = math.cos(yaw) * math.cos(pitch)
    front.y = math.sin(pitch)
    front.z = math.sin(yaw) * math.cos(pitch)-1
    cameraFront = glm.normalize(front)


def scroll_callback(window, xoffset, yoffset):
    global cameraPos
    cameraPos += glm.vec3(0, 0, yoffset)


def framebuffer_size_callback(window, width, height):
    GL_VIEWPORT(0, 0, width, height)


def processInput(window):
    global cameraPos
    if glfw.get_key(window, glfw.KEY_ESCAPE) == glfw.PRESS:
        glfw.set_window_should_close(window, True)

    cameraSpeed = 10 * deltaTime
    if glfw.get_key(window, glfw.KEY_W) == glfw.PRESS:
        cameraPos += (0, cameraSpeed, 0)
    if glfw.get_key(window, glfw.KEY_S) == glfw.PRESS:
        cameraPos += (0, -cameraSpeed, 0)
    if glfw.get_key(window, glfw.KEY_A) == glfw.PRESS:
        cameraPos += (-cameraSpeed, 0, 0)
    if glfw.get_key(window, glfw.KEY_D) == glfw.PRESS:
        cameraPos += (cameraSpeed, 0, 0)

if __name__ == '__main__':
    main()
