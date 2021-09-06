import math
import glfw


def get_coordinate(t, mod1, mod2):
    x = math.floor(t / mod1)
    y = math.floor((t - x * mod1) / mod2)
    z = (t - x * mod1 - y * mod2)
    return x, y, z


def processInput(window, deltaTime, cameraPos):
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
    return cameraPos
