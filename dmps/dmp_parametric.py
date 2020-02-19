from dmps.dmp import DMPs
import numpy as np


class DMPsParametric(DMPs):
    """A class used to represent a parametric DMP"""

    def __init__(self, n_dmps, n_bfs, minq, maxq, nbfs_q, **kwargs):

        self.minq = minq
        self.maxq = maxq
        self.n_bfsq = nbfs_q

        super().__init__(n_dmps, n_bfs, **kwargs)

        self.centers_t = None
        self.centers_q = None
        self.gen_centers()

        self.ht = np.ones(self.n_bfs) * self.n_bfs**1.5 / self.centers_t / self.cs.ax
        self.hq = kwargs.get("bfsq_variances", np.ones(self.n_bfsq))

    def gen_centers(self):

        des_ct = np.linspace(0, self.cs.run_time, self.n_bfs)

        self.centers_t = np.exp(-self.cs.ax * des_ct)
        self.centers_q = np.linspace(self.minq, self.maxq, self.n_bfsq)

    def gen_psi(self, x, q=None):

        t_psi = -self.ht * (x - self.centers_t) ** 2
        q_psi = -self.hq * (q - self.centers_q) ** 2

        return np.exp(np.add.outer(t_psi, q_psi))

    def generate_weights(self, w):
        if w is None:
            w = np.zeros((self.n_dmps, self.n_bfs, self.n_bfsq))
        self.w = w

    def gen_forcing_term(self, x, dmp_num, q=None):

        assert q is not None

        psi = self.gen_psi(x, q)
        return (self.gen_front_term(x, dmp_num) *
                np.sum((psi * self.w[dmp_num]).flatten()) / np.sum(psi.flatten()))



