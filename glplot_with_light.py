import ctypes

import OpenGL.GL.shaders
import numpy as np
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
from OpenGL.arrays import vbo

from PIL import Image

import glfw
import glm
from shader.shader import Shader
import math


from utils import vertex_info, util

model_path = './map/NewWorld1.obj_512.binvox'
cubes, cube_indices, textures_info = vertex_info.get_vertices_info(model_path)

scr_width, scr_height = 800, 600

cameraPos = glm.vec3(256, 256, 512)
deltaTime = 0
lastTime = 0

vertices = np.array([
        -0.5, -0.5, -0.5,
         0.5, -0.5, -0.5,
         0.5,  0.5, -0.5,
         0.5,  0.5, -0.5,
        -0.5,  0.5, -0.5,
        -0.5, -0.5, -0.5,

        -0.5, -0.5,  0.5,
         0.5, -0.5,  0.5,
         0.5,  0.5,  0.5,
         0.5,  0.5,  0.5,
        -0.5,  0.5,  0.5,
        -0.5, -0.5,  0.5,

        -0.5,  0.5,  0.5,
        -0.5,  0.5, -0.5,
        -0.5, -0.5, -0.5,
        -0.5, -0.5, -0.5,
        -0.5, -0.5,  0.5,
        -0.5,  0.5,  0.5,

         0.5,  0.5,  0.5,
         0.5,  0.5, -0.5,
         0.5, -0.5, -0.5,
         0.5, -0.5, -0.5,
         0.5, -0.5,  0.5,
         0.5,  0.5,  0.5,

        -0.5, -0.5, -0.5,
         0.5, -0.5, -0.5,
         0.5, -0.5,  0.5,
         0.5, -0.5,  0.5,
        -0.5, -0.5,  0.5,
        -0.5, -0.5, -0.5,

        -0.5,  0.5, -0.5,
         0.5,  0.5, -0.5,
         0.5,  0.5,  0.5,
         0.5,  0.5,  0.5,
        -0.5,  0.5,  0.5,
        -0.5,  0.5, -0.5
    ], dtype=np.float32)


def main():
    global deltaTime, lastTime, cameraPos
    if not glfw.init():
        raise Exception("GLFW can not be initialized.")

    window = glfw.create_window(scr_width, scr_height, "Model with Light", None, None)

    if not window:
        glfw.terminate()
        raise Exception("Failed to create GLFW window.")
    glfw.make_context_current(window)
    glfw.set_framebuffer_size_callback(window, framebuffer_size_callback)
    glfw.set_input_mode(window, glfw.CURSOR, glfw.CURSOR_DISABLED)
    glfw.set_scroll_callback(window, scroll_callback)

    glEnable(GL_DEPTH_TEST)
    glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)

    lightingShader = Shader('./glsl/light_casters.vs', './glsl/light_casters.fs')
    lightCubeShader = Shader('./glsl/normal_light_cube.vs', './glsl/normal_light_cube.fs')

    VBO = glGenBuffers(1)
    cubeVAO = glGenVertexArrays(1)
    EBO = glGenBuffers(1)

    glBindBuffer(GL_ARRAY_BUFFER, VBO)
    glBufferData(GL_ARRAY_BUFFER, cubes.itemsize * len(cubes), cubes, GL_STATIC_DRAW)
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO)
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, cube_indices.itemsize * len(cube_indices), cube_indices, GL_STATIC_DRAW)

    glBindVertexArray(cubeVAO)
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, cubes.itemsize*5, ctypes.c_void_p(0))
    glEnableVertexAttribArray(0)
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, cubes.itemsize*5, ctypes.c_void_p(cubes.itemsize*3))
    glEnableVertexAttribArray(1)

    _VBO = glGenBuffers(1)
    lightCubeVAO = glGenVertexArrays(1)
    glBindVertexArray(lightCubeVAO)
    glBindBuffer(GL_ARRAY_BUFFER, _VBO)
    glBufferData(GL_ARRAY_BUFFER, vertices.itemsize * len(cubes), vertices, GL_STATIC_DRAW)
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, vertices.itemsize*3, ctypes.c_void_p(0))
    glEnableVertexAttribArray(0)

    diffuseMap = loadTexture("res/textures/container2.png")
    specularMap = loadTexture("res/textures/container2_specular.png")

    lightingShader.use()
    lightingShader.set_int("material.diffuse", 0)
    lightingShader.set_int("material.specular", 1)



    while not glfw.window_should_close(window):
        currentTime = glfw.get_time()
        deltaTime = currentTime - lastTime
        lastTime = currentTime

        glClearColor(0, 0, 0, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        lightingShader.use()
        lightingShader.set_vec3("light.direction", -0.2, -1.0, -0.3)
        lightingShader.set_vec3("viewPos", cameraPos)

        lightingShader.set_vec3("light.ambient", 0.2, 0.2, 0.2)
        lightingShader.set_vec3("light.diffuse", 0.5, 0.5, 0.5)
        lightingShader.set_vec3("light.specular", 1.0, 1.0, 1.0)
        lightingShader.set_float("material.shininess", 32.0)

        model = glm.mat4(1.0)
        projection = glm.perspective(glm.radians(45), float(scr_width) / float(scr_height), 0.1, 512)
        lightingShader.set_mat4("projection", projection)
        lightingShader.set_mat4("model", model)
        lightingShader.set_mat4("t", model)



        cameraPos = util.processInput(window, deltaTime, cameraPos)
        print(cameraPos)

        view = glm.lookAt(cameraPos,
                          glm.vec3(256, 256, 0),
                          glm.vec3(0, 1, 0))
        lightingShader.setMat4("view", view)

        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, diffuseMap)

        glActiveTexture(GL_TEXTURE1)
        glBindTexture(GL_TEXTURE_2D, specularMap)

        glBindVertexArray(cubeVAO)
        glDrawElements(GL_QUADS, len(cube_indices), GL_UNSIGNED_INT, None)

        glfw.swap_buffers(window)
        glfw.poll_events()
    glDeleteVertexArrays(cubeVAO)
    glDeleteVertexArrays(lightCubeVAO)
    glDeleteBuffers(_VBO)
    glDeleteBuffers(EBO)
    glDeleteBuffers(VBO)
    glfw.terminate()

def scroll_callback(window, xoffset, yoffset):
    global cameraPos
    cameraPos += glm.vec3(0, 0, yoffset)


def framebuffer_size_callback(window, width, height):
    GL_VIEWPORT(0, 0, width, height)


def loadTexture(filepath):
    with Image.open(filepath) as im:
        textureID = glGenTextures(1)
        width, height, mode = im.size
        mode, data = im.mode, im.tobytes("raw", mode, 0, -1)

        if mode == "RGBA":
            mode = GL_RGBA
        elif mode == "RGB":
            mode = GL_RGB
        elif mode == "RED":
            mode = GL_RED

        glBindTexture(GL_TEXTURE_2D, textureID)
        glTexImage2D(GL_TEXTURE_2D, 0, mode, width, height, 0, mode, GL_UNSIGNED_BYTE, data)
        glGenerateMipmap(GL_TEXTURE_2D)

        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    return textureID
