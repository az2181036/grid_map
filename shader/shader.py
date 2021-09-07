import glm
from OpenGL.GL import *

vcode2 = """
#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aNormal;
layout (location = 2) in vec2 aTexCoords;

out vec3 FragPos;
out vec3 Normal;
out vec2 TexCoords;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

void main()
{
    FragPos = vec3(model * vec4(aPos, 1.0));
    Normal = aNormal;
    TexCoords = aTexCoords;

    gl_Position = projection * view * vec4(FragPos, 1.0);
}
"""


class Shader(object):
    def __init__(self, vertexfp, fragmentfp, geometryfp=None):
        with open(vertexfp) as vfp:
            vertexCode = vfp.read()
        vfp.close()

        with open(fragmentfp) as ffp:
            fragmentCode = ffp.read()
        ffp.close()

        if geometryfp is not None:
            with open(fragmentfp) as gfp:
                geometryCode = gfp.read()
            gfp.close()

        vShaderCode = str.encode(vertexCode)
        fShaderCode = str.encode(fragmentCode)

        vertex = glCreateShader(GL_VERTEX_SHADER)
        glShaderSource(vertex, vShaderCode)
        glCompileShader(vertex)
        self.__check_compile_errors(vertex, "VERTEX")

        fragment = glCreateShader(GL_FRAGMENT_SHADER)
        glShaderSource(fragment, fShaderCode)
        glCompileShader(fragment)
        self.__check_compile_errors(fragment, "FRAGMENT")

        if geometryfp is not None:
            gShaderCode = str.encode(geometryCode)
            geometry = glCreateShader(GL_GEOMETRY_SHADER)
            glShaderSource(geometry, gShaderCode)
            glCompileShader(geometry)
            self.__check_compile_errors(geometry, "GEOMETRY")

        self.ID = glCreateProgram()
        glAttachShader(self.ID, vertex)
        glAttachShader(self.ID, fragment)

        if geometryfp is not None:
            glAttachShader(self.ID, geometry)

        glLinkProgram(self.ID)
        self.__check_compile_errors(self.ID, "PROGRAM")
        glDeleteShader(vertex)
        glDeleteShader(fragment)
        if geometryfp is not None:
            glDeleteShader(geometry)

    def use(self):
        glUseProgram(self.ID)

    def set_bool(self, name, value):
        glUniform1i(glGetUniformLocation(self.ID, str.encode(name)), int(value))

    def set_int(self, name, value):
        glUniform1i(glGetUniformLocation(self.ID, str.encode(name)), value)

    def set_float(self, name, value):
        glUniform1f(glGetUniformLocation(self.ID, str.encode(name)), value)

    def set_vec2(self, name, value=None, vec=None):
        glUniform2fv(glGetUniformLocation(self.ID, str.encode(name)), 1, glm.value_ptr(value))

    def set_vec3(self, name, value):
        glUniform3fv(glGetUniformLocation(self.ID, str.encode(name)), 1, glm.value_ptr(value))

    def set_vec4(self, name, value):
        glUniform4fv(glGetUniformLocation(self.ID, str.encode(name)), 1, glm.value_ptr(value))

    def set_mat2(self, name, mat):
        glUniformMatrix2fv(glGetUniformLocation(self.ID, str.encode(name)), 1, GL_FALSE, glm.value_ptr(mat))

    def set_mat3(self, name, mat):
        glUniformMatrix3fv(glGetUniformLocation(self.ID, str.encode(name)), 1, GL_FALSE, glm.value_ptr(mat))

    def set_mat4(self, name, mat):
        glUniformMatrix4fv(glGetUniformLocation(self.ID, str.encode(name)), 1, GL_FALSE, glm.value_ptr(mat))


    def __check_compile_errors(self, shader, type):
        success = 0
        if type != "PROGRAM":
            success = glGetShaderiv(shader, GL_COMPILE_STATUS)
            if not success:
                infoLog = glGetShaderInfoLog(shader)
                print("ERROR::SHADER_COMPILATION_ERROR of type: ", type)
                print(infoLog)
                print("-- --------------------------------------------------- --")
        else:
            success = glGetProgramiv(shader, GL_LINK_STATUS)
            if not success:
                infoLog = glGetShaderInfoLog(shader)
                print("ERROR::PROGRAM_LINKING_ERROR of type: ", type)
                print(infoLog)
                print("-- --------------------------------------------------- --")