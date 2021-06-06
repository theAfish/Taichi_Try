from taichi_glsl import *

ti.init()


@ti.data_oriented
class WaveEquation:
    def __init__(self, resx, resy, lx, ly):
        self.resx = resx
        self.resy = resy
        self.lx = lx
        self.ly = ly
        self.dx = 1 / lx
        self.dy = 1 / ly
        self.a = 0.1
        self.dt = 1e-4
        # 0 for past, 1 for present
        self.u = ti.field(dtype=ti.f32, shape=(lx, ly, 2))
        self.u_draw = ti.field(dtype=ti.f32, shape=(lx, ly))

        # pre-calculated parameters
        self.a2 = self.a ** 2
        self.dx2 = self.dx ** 2
        self.dy2 = self.dy ** 2
        self.dt2 = self.dt ** 2
        self.at_x = self.a2 * self.dt2 / self.dx2
        self.at_y = self.a2 * self.dt2 / self.dy2

    @ti.kernel
    def init(self):
        # first for u(x,y,0)
        for x, y in ti.ndrange((127, 129),(127, 129)):
            self.u[x, y, 0] = 5
            self.u[x, y, 1] = 5
        # second for boundarys
        # u(0,y,t) & u(1,y,t)
        for y in range(self.ly):
            pass
            # self.u[0, y, 0] = 1
            # self.u[0, y, 1] = 1
            # self.T_field[self.lx - 1, y] = 0
        # u(x,0,t) & u(x,1,t)
        for x in range(self.lx):
            pass
            # if x > self.lx / 2:
            #     self.T_field[x, 0] = 1
            # else:
            #     self.T_field[x, self.ly - 1] = 1

    @ti.kernel
    def substep(self):
        for x, y in ti.ndrange((1, self.lx - 1), (1, self.ly - 1)):
            u_past = self.u[x, y, 0]
            self.u[x,y,0] = self.u[x,y,1]
            self.u[x, y, 1] = 2 * self.u[x,y,1] -2 * (self.at_x + self.at_y) * self.u[x, y, 1] \
                                  - u_past \
                                  + self.at_x * (self.u[x + 1, y, 1] + self.u[x - 1, y, 1]) \
                                  + self.at_y * (self.u[x, y + 1, 1] + self.u[x, y - 1, 1])

    @ti.kernel
    def write_grid(self):
        for x, y in ti.ndrange(self.lx, self.ly):
            self.u_draw[x,y] = self.u[x,y,1]

    def draw_canvas(self):
        gui = ti.GUI("Heat Field",
                     res=(self.resx, self.resy))
        self.init()
        while gui.running:
            for i in range(100):
                self.substep()
            self.write_grid()
            gui.set_image(ti.imresize(self.u_draw, self.resx, self.resy))
            gui.show()


if __name__ == "__main__":
    hf = WaveEquation(512, 512, 256, 256)
    hf.draw_canvas()
