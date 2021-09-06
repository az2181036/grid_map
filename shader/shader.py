from OpenGL.GL import *


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
                geometryCode = ffp.read()
            gfp.close()

        vShaderCode = str.encode(vertexCode)
        fShaderCode = str.encode(fragmentCode)

        vertex = glCreateShader(GL_VERTEX_SHADER)
        glShaderSource(vertex, 1, vShaderCode, None)
        glCompileShader(vertex)
        self.__check_compile_errors(vertex, "VERTEX")

        fragment = glCreateShader(GL_FRAGMENT_SHADER)
        glShaderSource(fragment, 1, fShaderCode, None)
        glCompileShader(fragment)
        self.__check_compile_errors(fragment, "FRAGMENT")

        if geometryfp is not None:
            gShaderCode = str.encode(geometryCode)
            geometry = glCreateShader(GL_GEOMETRY_SHADER)
            glShaderSource(geometry, 1, gShaderCode, None)
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
        if vec is not None:
            glUniform2f(glGetUniformLocation(self.ID, str.encode(name)), vec[0], vec[1])
        if value is not None:
            glUniform2fv(glGetUniformLocation(self.ID, str.encode(name)), 1, value[0])

    def set_vec3(self, name, value=None, vec=None):
        if vec is not None:
            glUniform3f(glGetUniformLocation(self.ID, str.encode(name)), vec[0], vec[1], vec[2])
        if value is not None:
            glUniform3fv(glGetUniformLocation(self.ID, str.encode(name)), 1, value[0])

    def set_vec4(self, name, value=None, vec=None):
        if vec is not None:
            glUniform4f(glGetUniformLocation(self.ID, str.encode(name)), vec[0], vec[1], vec[2], vec[3])
        if value is not None:
            glUniform4fv(glGetUniformLocation(self.ID, str.encode(name)), 1, value[0])

    def set_mat2(self, name, mat):
        glUniformMatrix2fv(glGetUniformLocation(self.ID, str.encode(name)), 1, GL_FALSE, mat[0][0])

    def set_mat3(self, name, mat):
        glUniformMatrix3fv(glGetUniformLocation(self.ID, str.encode(name)), 1, GL_FALSE, mat[0][0])

    def set_mat4(self, name, mat):
        glUniformMatrix4fv(glGetUniformLocation(self.ID, str.encode(name)), 1, GL_FALSE, mat[0][0])


    def __check_compile_errors(self, shader, type):
        success = 0
        infoLog = []
        if type != "PROGRAM":
            glGetShaderiv(shader, GL_COMPILE_STATUS, success)
            if not success:
                glGetShaderInfoLog(shader, 1024, None, infoLog)
                print("ERROR::SHADER_COMPILATION_ERROR of type: ", type)
                print(infoLog)
                print("-- --------------------------------------------------- --")
        else:
            glGetProgramiv(shader, GL_LINK_STATUS, success)
            if not success:
                glGetShaderInfoLog(shader, 1024, None, infoLog)
                print("ERROR::PROGRAM_LINKING_ERROR of type: ", type)
                print(infoLog)
                print("-- --------------------------------------------------- --")