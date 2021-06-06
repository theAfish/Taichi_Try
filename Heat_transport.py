from taichi_glsl import *

ti.init()


@ti.data_oriented
class HeatField:
    def __init__(self, resx, resy, lx, ly):
        self.resx = resx
        self.resy = resy
        self.lx = lx
        self.ly = ly
        self.dx = 1 / lx
        self.dy = 1 / ly
        self.a = 0.1
        self.dt = 1e-3
        self.T_field = ti.field(dtype=ti.f32, shape=(lx, ly))
        self.paused = ti.field(dtype=ti.i32, shape=())

        # pre-calculated parameters
        self.a2 = self.a ** 2
        self.dx2 = self.dx ** 2
        self.dy2 = self.dy ** 2
        self.at_x = self.a2 * self.dt / self.dx2
        self.at_y = self.a2 * self.dt / self.dy2

    @ti.kernel
    def init(self):
        # first for u(x,y,0)
        # for x, y in ti.ndrange((self.lx/4, 3*self.lx/4),(self.ly/4, 3*self.ly/4)):
        #     self.T_field[x,y] = 1
        # second for boundarys
        # u(0,y,t) & u(1,y,t)
        for y in range(self.ly):
            self.T_field[0, y] = 0
            self.T_field[self.lx - 1, y] = 1
        # u(x,0,t) & u(x,1,t)
        for x in range(self.lx):
            if x > self.lx / 2:
                self.T_field[x, 0] = 1
            else:
                self.T_field[x, self.ly - 1] = 1

    @ti.kernel
    def substep(self):
        for x, y in ti.ndrange((1, self.lx - 1), (1, self.ly - 1)):
            self.T_field[x, y] += -2 * (self.at_x + self.at_y) * self.T_field[x, y] \
                                  + self.at_x * (self.T_field[x + 1, y] + self.T_field[x - 1, y]) \
                                  + self.at_y * (self.T_field[x, y + 1] + self.T_field[x, y - 1])

    def draw_canvas(self):
        gui = ti.GUI("Heat Field",
                     res=(self.resx, self.resy))
        self.init()
        while gui.running:
            for e in gui.get_events(ti.GUI.PRESS):
                if e.key == gui.SPACE:
                    self.paused[None] = not self.paused[None]
            if self.paused[None]:
                for i in range(10):
                    self.substep()
            gui.set_image(ti.imresize(self.T_field, self.resx, self.resy))
            gui.show()


if __name__ == "__main__":
    hf = HeatField(512, 512, 128, 128)
    hf.draw_canvas()
