# Shadertoy "Fractal Tiling", reference ==> https://www.shadertoy.com/view/Ml2GWy#

import taichi as ti
import taichi.math as m

ti.init(arch=ti.cpu)

res_x = 768
res_y = 512
pixels = ti.Vector.field(3, ti.f32, shape=(res_x, res_y))


@ti.kernel
def render(t: ti.f32):
    for i_, j_ in pixels:
        c = 0.0
        tile_size = 7

        offset = int(t*5)  # make it move
        for k in range(0, 6):
            weight = (k + 0.6) / 17
            ts = tile_size * 2 ** k
            i = (i_ + offset) // ts
            j = (j_ + offset) // ts
            center = ti.Vector([i, j]) * ts + ts / 2.0
            r = (ti.Vector([i_ + offset, j_ + offset]) - center).norm()*1.2 / ts
            c += m.fract(m.sin((i * 42 + j * 8 + i * j * 1 + 457.6) * 1000.0)
                         * m.clamp(m.cos((t * 0.05 + i * 31 + j * 42 + 991) * 0.5) * 1.2 + 1.0, 0.0, 1.0)) * weight * m.smoothstep(1.0, 0.3, r)
            pass
        color = ti.Vector([c, c, c]) * ti.Vector([199, 111, 139]) / 255.0 * 1.8
        pixels[i_, j_] = color


window = ti.ui.Window("Fractal Tiling", res=(res_x, res_y))
canvas = window.get_canvas()
i = 0

while window.running:
    render(i*0.05)
    canvas.set_image(pixels)
    window.show()
    i += 1
