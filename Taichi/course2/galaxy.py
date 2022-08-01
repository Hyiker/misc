from time import time
import taichi as ti
from celestial_objects import Star, Planet, SuperStar


def get_time() -> float:
    return float(time())


if __name__ == "__main__":
    ti.init(arch=ti.cuda)

    # control
    paused = False
    export_images = False

    # stars and planets
    stars = Star(N=2, mass=1000)
    stars.initialize(0.5, 0.5, 0.2, 10)
    planets = Planet(N=1000, mass=1)
    planets.initialize(0.5, 0.5, 0.4, 10)
    superstars = SuperStar(N=1, mass=5000)
    superstars.initialize(0.5, 0.5, 0.2, 0)

    # GUI
    window = ti.ui.Window("Galaxy", (800, 800))
    canvas = window.get_canvas()
    canvas.set_background_color((0.0, 0.0, 0.0))
    h = 5e-5  # time-step size
    i = 0
    fu_fps = 60.0
    fu_gap = 1 / fu_fps
    t_last = get_time()
    first = True
    while window.running:
        if first:
            t_last = get_time()
            first = False
            continue
        t = get_time()
        need_recompute = False
        if t - t_last >= fu_gap:
            need_recompute = True
            t_last = get_time()
        for e in window.get_events(ti.ui.PRESS):
            if e.key == ti.ui.ESCAPE:
                exit()
            elif e.key == ti.ui.SPACE:
                paused = not paused
                print("paused =", paused)
            elif e.key == ti.ui.UP:
                stars.add_mass(200)
                print("add star mass to {}".format(stars.m[None]))
            elif e.key == ti.ui.DOWN:
                stars.add_mass(-200)
                print("reduce star mass to {}".format(stars.m[None]))
            elif e.key == 'r':
                stars.initialize(0.5, 0.5, 0.2, 10)
                planets.initialize(0.5, 0.5, 0.4, 10)
                i = 0
            elif e.key == 'i':
                export_images = not export_images

        if not paused and need_recompute:
            stars.clearForce()
            stars.computeForce()
            superstars.clearForce()
            superstars.computeForce()
            planets.clearForce()
            planets.computeForce(stars)
            planets.computeForce(superstars)
            for celestial_obj in (stars, planets, superstars):
                celestial_obj.update(h)
            i += 1

        planets.display(canvas)
        stars.display(canvas, radius=10, color=0xffd500)
        superstars.display(canvas, radius=20, color=0x2a23ad)
        if export_images:
            window.write_image(f"images\output_{i:05}.png")
        else:
            window.show()
