from dmps.dmp import DMPs
import numpy as np


class DMPsDiscrete(DMPs):
    """A class used to represent a discrete DMP"""

    def __init__(self, **kwargs):

        super().__init__(**kwargs)

        self.centers = None
        self.gen_centers()
        self.h = np.ones(self.n_bfs) * self.n_bfs ** 1.5 / self.centers / self.cs.ax

        self.check_offset()

    def gen_centers(self):
        des_c = np.linspace(0, self.cs.run_time, self.n_bfs)
        self.centers = np.exp(-self.cs.ax * des_c)

    def gen_psi(self, x):
        return np.exp(-self.h * (x - self.centers)**2)

    def initialize_weights(self, w):
        if w is None:
            w = np.zeros((self.n_dmps, self.n_bfs))
        self.w = w

    def gen_forcing_term(self, x, d, **kwargs):
        psi = self.gen_psi(x)
        return (self.gen_front_term(x, d) *
                (np.dot(psi, self.w[d])) / np.sum(psi))

def main():
    dmp = DMPsDiscrete(y0 = -20)
    y, dy, ddy = dmp.rollout(forcing_term = False)
    print(y)
    import matplotlib.pyplot as plt
    plt.plot(y)
    plt.xlabel("time")
    plt.ylabel("trajectory")
    plt.show()


if __name__ == "__main__":
    main()
