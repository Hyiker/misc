from taichi.math import ivec2, normalize, vec2, vec3, vec4
import matplotlib.pyplot as plt
import taichi as ti

ti.init(arch=ti.gpu)
SURFEL_MOMENT_RESOLUTION = 4
SURFEL_MOMENT_TEXELS = 1 + SURFEL_MOMENT_RESOLUTION + 1
THREADCOUNT = 8

dim = SURFEL_MOMENT_RESOLUTION

res = ti.Vector.field(n=3, dtype=float, shape=(dim, dim))


@ti.func
def encode_hemioct(v):
    p = v.xy * (1.0 / (abs(v.x) + abs(v.y) + v.z))
    return vec2(p.x + p.y, p.x - p.y)


@ti.func
def decode_hemioct(e):
    temp = vec2(e.x + e.y, e.x - e.y) * 0.5
    v = vec3(temp, 1.0 - abs(temp.x) - abs(temp.y))
    return normalize(v)


@ti.func
def compute_texel_direction(xy):
    return vec2(xy + 0.5) / vec2(dim) * 2 - 1


@ti.kernel
def get_basis():
    for i, j in res:
        res[i, j] = normalize(decode_hemioct(compute_texel_direction(ivec2(i, j))))


get_basis()
res_np = res.to_numpy()

fig = plt.figure()
ax = fig.add_subplot(projection="3d")
ax.axes.set_xlim3d(left=-1.0, right=1.0)
ax.axes.set_ylim3d(bottom=-1.0, top=1.0)
ax.axes.set_zlim3d(bottom=0.0, top=1.0)

for i in range(dim):
    for j in range(dim):
        ax.scatter(res_np[i][j][0], res_np[i][j][1], res_np[i][j][2])

plt.show()
