import ctypes

import OpenGL.GL.shaders
import numpy as np
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *

from PIL import Image

import glfw
import glm
from shader.shader import Shader
import math


from utils import vertex_info, util

vertices = np.array([
    -0.5, -0.5, -0.5,
    0.5, -0.5, -0.5,
    0.5, 0.5, -0.5,
    0.5, 0.5, -0.5,
    -0.5, 0.5, -0.5,
    -0.5, -0.5, -0.5,

    -0.5, -0.5, 0.5,
    0.5, -0.5, 0.5,
    0.5, 0.5, 0.5,
    0.5, 0.5, 0.5,
    -0.5, 0.5, 0.5,
    -0.5, -0.5, 0.5,

    -0.5, 0.5, 0.5,
    -0.5, 0.5, -0.5,
    -0.5, -0.5, -0.5,
    -0.5, -0.5, -0.5,
    -0.5, -0.5, 0.5,
    -0.5, 0.5, 0.5,

    0.5, 0.5, 0.5,
    0.5, 0.5, -0.5,
    0.5, -0.5, -0.5,
    0.5, -0.5, -0.5,
    0.5, -0.5, 0.5,
    0.5, 0.5, 0.5,

    -0.5, -0.5, -0.5,
    0.5, -0.5, -0.5,
    0.5, -0.5, 0.5,
    0.5, -0.5, 0.5,
    -0.5, -0.5, 0.5,
    -0.5, -0.5, -0.5,

    -0.5, 0.5, -0.5,
    0.5, 0.5, -0.5,
    0.5, 0.5, 0.5,
    0.5, 0.5, 0.5,
    -0.5, 0.5, 0.5,
    -0.5, 0.5, -0.5], dtype=np.float32)

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

def scroll_callback(window, xoffset, yoffset):
    global cameraPos
    cameraPos += glm.vec3(0, 0, yoffset)


def framebuffer_size_callback(window, width, height):
    GL_VIEWPORT(0, 0, width, height)

deltaTime, lastTime = 0,0
scr_width, scr_height = 800, 600
cameraPos = glm.vec3(0, 0, 3)
glfw.init()
window = glfw.create_window(scr_width, scr_height, "Model with Light", None, None)

glfw.make_context_current(window)
glfw.set_framebuffer_size_callback(window, framebuffer_size_callback)
glfw.set_input_mode(window, glfw.CURSOR, glfw.CURSOR_DISABLED)
glfw.set_scroll_callback(window, scroll_callback)


shader = OpenGL.GL.shaders.compileProgram(OpenGL.GL.shaders.compileShader(vertex_shader, GL_VERTEX_SHADER),
                                              OpenGL.GL.shaders.compileShader(fragment_shader, GL_FRAGMENT_SHADER))

VAO = glGenVertexArrays(1)
VBO = glGenBuffers(1)

glBindBuffer(GL_ARRAY_BUFFER, VBO)
glBufferData(GL_ARRAY_BUFFER, vertices.itemsize*len(vertices), vertices, GL_STATIC_DRAW)
glBindVertexArray(VAO)
glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, vertices.itemsize * 3, ctypes.c_void_p(0))
glEnableVertexAttribArray(0)

glUseProgram(shader)
glEnable(GL_DEPTH_TEST)

while not glfw.window_should_close(window):
    currentTime = glfw.get_time()
    deltaTime = currentTime - lastTime
    lastTime = currentTime

    cameraPos = util.processInput(window, deltaTime, cameraPos)

    glClearColor(1, 1, 1, 1.0)
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    glUseProgram(shader)
    model = glm.mat4(1.0)
    projection = glm.perspective(glm.radians(45), float(scr_width) / float(scr_height), 0.1, 100)
    view = glm.lookAt(cameraPos,
                      glm.vec3(0, 0, 0),
                      glm.vec3(0, 1, 0))

    view_loc = glGetUniformLocation(shader, "view")
    proj_loc = glGetUniformLocation(shader, "projection")
    model_loc = glGetUniformLocation(shader, "model")

    glUniformMatrix4fv(model_loc, 1, GL_FALSE, glm.value_ptr(model))
    glUniformMatrix4fv(proj_loc, 1, GL_FALSE, glm.value_ptr(projection))
    glUniformMatrix4fv(view_loc, 1, GL_FALSE, glm.value_ptr(view))

    glBindVertexArray(VAO)
    glDrawArrays(GL_TRIANGLES, 0, 36)

    glfw.swap_buffers(window)
    glfw.poll_events()

glDeleteVertexArrays(VAO)
glDeleteBuffers(VBO)
glfw.terminate()

