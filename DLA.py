from taichi_glsl import *

ti.init()
RES = 512
substeps = 10
lattice_size = 150
num_walkers = 200
num_neighbours = 4

walkers = ti.Vector.field(2, dtype=ti.i32, shape=num_walkers)
grid = ti.field(dtype=ti.i32, shape=(lattice_size, lattice_size))
canvas = ti.Vector.field(3, dtype=ti.f32, shape=(lattice_size, lattice_size))
neighbours = ti.Vector.field(2, dtype=ti.i32, shape=4)
neighbours.from_numpy(np.array([[0, 1], [1, 0], [-1, 0], [0, -1]]))

@ti.kernel
def init():
    grid[lattice_size/2, lattice_size/2] = 1
    for i in walkers:
        random_set_walker(i)

@ti.func
def random_set_walker(i):
    p1 = randInt(0, lattice_size - 1)
    p2 = randInt(0, 2) * (lattice_size - 1)
    if ti.random() < 0.5:
        walkers[i] = [p1, p2]
    else:
        walkers[i] = [p2, p1]

@ti.func
def random_walk(i):
    n = neighbours[randInt(0, 4)]
    inter = n + walkers[i]
    if (inter[0] < 0 or inter[0] >= lattice_size) or (inter[1] < 0 or inter[1] >= lattice_size):
        walkers[i] += -n
    else:
        walkers[i] += n

@ti.kernel
def growth():
    for i in walkers:
        for n in range(num_neighbours):
            nei = neighbours[n] + walkers[i]
            if grid[nei] == 1:
                grid[walkers[i]] = 1
                random_set_walker(i)
            else:
                random_walk(i)

@ti.kernel
def set_canvas():
    for i, j in grid:
        canvas[i, j] = [0.01, 0.01, 0.1]
    for i, j in grid:
        if grid[i, j] == 1:
            canvas[i, j] += [0.8, 0.4, 0.8]
            canvas[i, j + 1] += [0.2, 0.1, 0.2]
            canvas[i, j - 1] += [0.2, 0.1, 0.2]
            canvas[i + 1, j] += [0.2, 0.1, 0.2]
            canvas[i - 1, j] += [0.2, 0.1, 0.2]
    for i in walkers:
        canvas[walkers[i]] = [0.5, 0.1, 0.5]


def main():
    gui = ti.GUI(name="Diffusion-Limited Aggregation", res=RES)
    init()

    while gui.running:
        for _ in range(substeps):
            growth()
        set_canvas()

        gui.set_image(ti.imresize(canvas, RES))
        gui.show()

if __name__ == "__main__":
    main()
