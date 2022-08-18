import random
from time import time
import taichi as ti
import taichi.math as tm

ti.init(arch=ti.cuda)
pixels = ti.Vector.field(3, dtype=ti.f32, shape=(512, 512))
gsize = 32
grid = ti.field(ti.i32, shape=(512 // gsize, 512 // gsize, 2))
grid[3, 3, 0] = 3

pingpong = ti.field(ti.i32, shape=())
direction = ti.Vector([0, -1])
regenerate_food = ti.field(ti.i32, shape=())
alive = ti.field(ti.u8, shape=())
alive[None] = 1


@ti.data_oriented
class Snake:
    def __init__(self, i, j):
        self.body = ti.Vector.field(2, ti.i32, shape=(16 * 16, 2))
        self.length = ti.field(ti.i32, shape=())
        self.length[None] = 1
        self.body[0, 0] = ti.Vector([i, j])

    @ti.kernel
    def update(self, direction: ti.types.vector(2, ti.i32)):
        for i in range(self.length[None]):
            now = pingpong[None]
            last = now ^ 1
            if i == 0:
                self.body[0, now] = self.body[0, last] + direction
                n = self.body[0, now]
                if grid[n[0], n[1], last] == 2:
                    alive[None] = 0
                elif grid[n[0], n[1], now] == 3:
                    self.body[self.length[None], now] = self.body[self.length[None] - 1, last]
                    self.length[None] += 1
                    regenerate_food[None] = 1
            else:
                self.body[i, now] = self.body[i - 1, last]
            self.body[i, now] = (self.body[i, now] + 16) % 16

    @ti.func
    def isbody(self, i, j, index):
        ret = 0
        for k in range(self.length[None]):
            if all(snake.body[k, index] == ti.Vector([i, j])):
                ret = 2 if k == 0 else 1
                break
        return ret


headcolor = ti.Vector([243, 75, 125]) / 255.0
bodycolor = ti.Vector([241, 224, 90]) / 255.0
foodcolor = ti.Vector([48, 161, 78]) / 255.0

pingpong[None] = 0
snake = Snake(8, 8)


@ti.kernel
def tick():
    for i, j in ti.ndrange(grid.shape[0], grid.shape[1]):
        now = pingpong[None]
        last = now ^ 1
        b = snake.isbody(i, j, now)
        l = grid[i, j, last]
        n = l
        if b == 2:
            n = 1
        elif b == 1:
            n = 2
        elif l != 3:
            n = 0
        grid[i, j, now] = n


@ti.kernel
def render():
    for i, j in pixels:
        i_, j_ = i // gsize, j // gsize
        center = ti.Vector([i_, j_]) * gsize + gsize // 2
        pix = ti.Vector([i, j])
        vec = ti.abs(pix - center)
        c = ti.Vector([0.0, 0.0, 0.0])
        index = pingpong[None]
        if vec[0] < gsize * 0.4 and vec[1] < gsize * 0.4:
            if grid[i_, j_, index] == 0:
                c += 1.0
            elif grid[i_, j_, index] == 1:
                c = headcolor
            elif grid[i_, j_, index] == 2:
                c = bodycolor
            elif grid[i_, j_, index] == 3:
                c = foodcolor
        pixels[i, j] = c


window = ti.ui.Window('Snake Game', (512, 512), vsync=True)
canvas = window.get_canvas()
time_gap = .3
last_time = -1.0
confirm_dir = False
if __name__ == '__main__':
    while window.running:
        if not confirm_dir:
            if window.is_pressed(ti.ui.UP, 'w') and direction[1] == 0:
                confirm_dir = True
                direction = ti.Vector([0, 1])
            elif window.is_pressed(ti.ui.DOWN, 's') and direction[1] == 0:
                confirm_dir = True
                direction = ti.Vector([0, -1])
            elif window.is_pressed(ti.ui.LEFT, 'a') and direction[0] == 0:
                confirm_dir = True
                direction = ti.Vector([-1, 0])
            elif window.is_pressed(ti.ui.RIGHT, 'd') and direction[0] == 0:
                confirm_dir = True
                direction = ti.Vector([1, 0])
        if last_time < 0 or time() - last_time >= time_gap:
            confirm_dir = False
            pingpong[None] = 1 ^ pingpong[None]
            snake.update(direction)
            if not alive[None]:
                print("game over")
                exit(0)
            tick()
            if regenerate_food[None]:
                i, j = random.randint(0, 15), random.randint(0, 15)
                while grid[i, j, pingpong[None]] != 0:
                    i, j = random.randint(0, 15), random.randint(0, 15)
                grid[i, j, pingpong[None]] = 3
                regenerate_food[None] = 0
            last_time = time()
        render()
        canvas.set_image(pixels)
        window.show()
