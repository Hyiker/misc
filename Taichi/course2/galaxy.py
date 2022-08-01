import taichi as ti
from celestial_objects import Star, Planet, SuperStar

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
    my_gui = ti.GUI("Galaxy", (800, 800))
    h = 5e-5  # time-step size
    i = 0
    while my_gui.running:

        for e in my_gui.get_events(ti.GUI.PRESS):
            if e.key == ti.GUI.ESCAPE:
                exit()
            elif e.key == ti.GUI.SPACE:
                paused = not paused
                print("paused =", paused)
            elif e.key == ti.GUI.UP:
                stars.add_mass(200)
                print("add star mass to {}".format(stars.m[None]))
            elif e.key == ti.GUI.DOWN:
                stars.add_mass(-200)
                print("reduce star mass to {}".format(stars.m[None]))
            elif e.key == 'r':
                stars.initialize(0.5, 0.5, 0.2, 10)
                planets.initialize(0.5, 0.5, 0.4, 10)
                i = 0
            elif e.key == 'i':
                export_images = not export_images

        if not paused:
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

        stars.display(my_gui, radius=10, color=0xffd500)
        superstars.display(my_gui, radius=20, color=0xff00d5)
        planets.display(my_gui)
        if export_images:
            my_gui.show(f"images\output_{i:05}.png")
        else:
            my_gui.show()
