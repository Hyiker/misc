import taichi as ti
import numpy as np
from time import time
ti.init(arch=ti.cuda, debug=True)

window_size = (512, 512)

pixels = ti.field(ti.f32)
# hierarchical layout
hl = ti.root.dense(ti.i, window_size[0]).dense(ti.j, window_size[1])
# flat layout
# fl = ti.root.dense(ti.i, 400).dense(ti.j, 400)
gs = 64.0
hl.place(pixels)

# len = 256
ptable = ti.field(ti.i32, shape=512)
ptable.from_numpy(np.array([151, 160, 137, 91, 90, 15, 131, 13, 201, 95, 96, 53, 194, 233, 7,
                            225, 140, 36, 103, 30, 69, 142, 8, 99, 37, 240, 21, 10, 23, 190, 6, 148, 247,
                            120, 234, 75, 0, 26, 197, 62, 94, 252, 219, 203, 117, 35, 11, 32, 57, 177, 33,
                            88, 237, 149, 56, 87, 174, 20, 125, 136, 171, 168, 68, 175, 74, 165, 71, 134,
                            139, 48, 27, 166, 77, 146, 158, 231, 83, 111, 229, 122, 60, 211, 133, 230, 220,
                            105, 92, 41, 55, 46, 245, 40, 244, 102, 143, 54, 65, 25, 63, 161, 1, 216, 80,
                            73, 209, 76, 132, 187, 208, 89, 18, 169, 200, 196, 135, 130, 116, 188, 159, 86,
                            164, 100, 109, 198, 173, 186, 3, 64, 52, 217, 226, 250, 124, 123, 5, 202, 38,
                            147, 118, 126, 255, 82, 85, 212, 207, 206, 59, 227, 47, 16, 58, 17, 182, 189,
                            28, 42, 223, 183, 170, 213, 119, 248, 152, 2, 44, 154, 163, 70, 221, 153, 101,
                            155, 167, 43, 172, 9, 129, 22, 39, 253, 19, 98, 108, 110, 79, 113, 224, 232,
                            178, 185, 112, 104, 218, 246, 97, 228, 251, 34, 242, 193, 238, 210, 144, 12,
                            191, 179, 162, 241, 81, 51, 145, 235, 249, 14, 239, 107, 49, 192, 214, 31, 181,
                            199, 106, 157, 184, 84, 204, 176, 115, 121, 50, 45, 127, 4, 150, 254, 138, 236,
                            205, 93, 222, 114, 67, 29, 24, 72, 243, 141, 128, 195, 78, 66, 215, 61, 156, 180] * 2))

grad_table = ti.Vector.field(2, ti.f32, shape=8)
grad_table.from_numpy(np.array([(1, 2), (2, 1), (-1, 2), (-2, 1), (-1, -2), (-2, -1), (1, -2), (2, -1)]))
minn = ti.field(ti.f32, shape=())


@ti.func
def smooth(t):
    return 6.0 * ti.pow(t, 5.0) - 15.0 * ti.pow(t, 4.0) + 10.0 * ti.pow(t, 3.0)


@ti.func
def lerp(v1, v2, t):
    g = 1.0 - smooth(t)
    return v1 * g + v2 * (1.0 - g)


@ti.func
def lerp2D(v00, v01, v10, v11, tu, tv):
    v0y = lerp(v00, v01, tu)
    v1y = lerp(v10, v11, tu)
    return lerp(v0y, v1y, tv)


@ti.func
def grad(U, V):
    s = ptable[(ptable[U & 255] + V) & 255]
    return grad_table[s & 7]


@ti.func
def dot_prod(u, v, ug, vg):
    delta = ti.Vector([u - ug, v - vg])
    grad = grad(ug, vg).normalized()
    return ti.math.dot(delta, grad)


@ti.func
def perlin_noise(x, y, z):
    u = (x + z) / gs
    v = (y + z) / gs
    u0 = ti.floor(u, ti.i32)
    v0 = ti.floor(v, ti.i32)
    u1 = u0 + 1
    v1 = v0 + 1

    prod00 = dot_prod(u, v, u0, v0)
    prod01 = dot_prod(u, v, u1, v0)
    prod10 = dot_prod(u, v, u0, v1)
    prod11 = dot_prod(u, v, u1, v1)

    result = lerp2D(prod00, prod01, prod10, prod11, u - u0, v - v0)
    return (result + 0.707) / 1.414


@ti.kernel
def render(t: ti.f32):
    for x, y in pixels:
        pixels[x, y] = perlin_noise(x, y, t)


if __name__ == '__main__':
    window = ti.ui.Window('Perlin Noise', window_size)
    canvas = window.get_canvas()
    t = 0.0
    last_time = float(time())
    while window.running:
        now_time = float(time())
        delta_time = now_time - last_time
        t += delta_time
        last_time = now_time
        render(t * 100.0)
        canvas.set_image(pixels)
        window.show()