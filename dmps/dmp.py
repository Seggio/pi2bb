import numpy as np
from dmps.cs import CanonicalSystem


class DMP:
    """
       A class used to represent a discrete DMP

       Attributes
       ----------
       n_dof : int
           number of degrees of freedom
       n_bfs : int
           number of basis functions, same for each DoF
       y0: int/np.array
            initial state of the DMP
       g: int/np.array
            DMP goal
       w: np.array
            DMP weights
       y/dy/ddy: np.array
            Current Position/Velocity/Acceleration

       Methods
       -------
       rollout(timesteps=None)
           Executes a rollout of the DMPs with a specific number of timesteps

       step()
           Executes a single DMP step
       """

    def __init__(self, n_dof=1, n_bfs=25,
                 dt=.01, run_time=1.0,
                 w=None, y0=0, g=1, dy0=0,
                 ay=None, by=None, **kwargs):

        self.n_dof = n_dof
        self.n_bfs = n_bfs

        self.dt = dt
        self.run_time = run_time

        if isinstance(y0, (int, float)):
            y0 = np.ones(self.n_dof) * y0
        self.y0 = y0

        if isinstance(dy0, (int, float)):
            dy0 = np.ones(self.n_dof) * dy0
        self.dy0 = dy0

        if isinstance(g, (int, float)):
            g = np.ones(self.n_dof) * g
        self.g = g

        if w is None:
            w = np.zeros((self.n_dof, self.n_bfs))
        self.w = w

        # Default value for ay/by to have the system critically dumped
        self.ay = np.ones(self.n_dof) * 25. if ay is None else ay
        self.by = self.ay / 4. if by is None else by

        # set up the CS
        self.cs = CanonicalSystem(dt=self.dt, run_time=self.run_time, **kwargs)

        # set up the kernels
        self.centers = None
        self.gen_centers()
        self.h = np.ones(self.n_bfs) * self.n_bfs ** 1.5 / self.centers / self.cs.ax

        # set up the initial DMP state
        self.y = None
        self.dy = None
        self.ddy = None
        self.reset_state()

    def gen_front_term(self, x, d):
        """Generates the term x*(goal - y0) for a specific DoF """
        return x * (self.g[d] - self.y0[d])

    def gen_activations(self, x):
        if isinstance(x, np.ndarray):
            x = x[:, None]
        return np.exp(-self.h * (x - self.centers) ** 2)

    def gen_centers(self):
        des_c = np.linspace(0, self.cs.run_time, self.n_bfs)
        self.centers = np.exp(-self.cs.ax * des_c)

    def gen_forcing_term(self, x, d):
        psi = self.gen_activations(x)
        return (self.gen_front_term(x, d) *
                (np.dot(psi, self.w[d])) / np.sum(psi))

    def rollout(self, tau=1.0):
        """Executes a complete rollout of the DMPs

        Retruns:
        y_track: trajectory array
        dy_track: velocitis array
        ddy_track: accelerations array"""

        self.reset_state()

        n_steps = self.compute_nsteps()

        y_track = np.zeros((n_steps, self.n_dof))
        dy_track = np.zeros((n_steps, self.n_dof))
        ddy_track = np.zeros((n_steps, self.n_dof))

        y_track[0] = self.y0
        dy_track[0] = self.dy0
        y_track[1] = y_track[0] + dy_track[0] * tau * self.dt

        for t in range(n_steps-2):
            y_track[t+2], dy_track[t+1], ddy_track[t] = self.step(tau=tau)
        _, dy_track[n_steps - 1], ddy_track[n_steps - 2] = self.step(tau=tau)
        _, _, ddy_track[n_steps-1] = self.step(tau=tau)

        return y_track, dy_track, ddy_track

    def step(self, tau=1.0):
        """Executes a single DMP step"""

        # single step of the canonical system
        x = self.cs.step(tau=tau)

        # step for each DoF
        for d in range(self.n_dof):

            # generate the forcing term
            f = self.gen_forcing_term(x, d)

            # Current DMP acceleration, velocity, position
            self.ddy[d] = (self.ay[d] *
                           (self.by[d] * (self.g[d] - self.y[d]) -
                           self.dy[d]) + f)

            self.dy[d] += self.ddy[d] * tau * self.dt
            self.y[d] += self.dy[d] * tau * self.dt

        return self.y, self.dy, self.ddy

    def reset_state(self):

        self.y = self.y0.copy()
        self.dy = self.dy0.copy()
        self.ddy = np.zeros(self.n_dof)

        self.cs.reset_state()

    def check_offset(self):
        """Checks the offsest g-y0, if 0 adds a noise to g in order to have a forcing term != 0"""

        for d in range(self.n_dof):
            if self.y0[d] == self.g[d]:
                self.g[d] += 1e-4

    def get_params(self):
        """Returns a dictionary {param_name: param_value}"""

        return {"y0": self.y0,
                "g": self.g,
                "w": self.w}

    def imitate_path(self, y_des, plot=False):

        # set initial state and goal
        if y_des.ndim == 1:
            y_des = y_des.reshape(1, len(y_des))
        assert y_des.ndim == 2

        # Setting starting position and goal
        self.y0 = y_des[:, 0].copy()
        self.g = y_des[:, -1].copy()

        self.check_offset()

        # generate function to interpolate the desired trajectory
        import scipy.interpolate

        n_steps = self.compute_nsteps()
        path = np.zeros((self.n_dof, n_steps))
        x = np.linspace(0, self.run_time, y_des.shape[1])

        for d in range(self.n_dof):
            path_gen = scipy.interpolate.interp1d(x, y_des[d])
            for t in range(n_steps):
                path[d, t] = path_gen(t * self.dt)
        y_des = path

        # calculate velocity of y_des with central differences
        dy_des = np.gradient(y_des, axis=1) / self.dt

        # calculate acceleration of y_des with central differences
        ddy_des = np.gradient(dy_des, axis=1) / self.dt

        f_target = np.zeros((y_des.shape[1], self.n_dof))
        # find the force required to move along this trajectory
        for d in range(self.n_dof):
            f_target[:, d] = ddy_des[d] - self.ay[d] * (
                    self.by[d] * (self.g[d] - y_des[d]) - dy_des[d]
            )

        # calculate x and psi
        x_track = self.cs.rollout()
        psi_track = self.gen_activations(x_track)

        # efficiently calculate BF weights using weighted linear regression
        self.w = np.zeros((self.n_dof, self.n_bfs))
        for d in range(self.n_dof):
            # spatial scaling term
            k = self.g[d] - self.y0[d]
            for b in range(self.n_bfs):
                num = np.sum(x_track * psi_track[:, b] * f_target[:, d])
                den = np.sum(x_track ** 2 * psi_track[:, b])
                self.w[d, b] = num / den
                if abs(k) > 1e-5:
                    self.w[d, b] /= k

        self.w = np.nan_to_num(self.w)

        self.reset_state()
        return y_des

    def compute_nsteps(self):
        return int(self.run_time / self.dt)


def main():

    import matplotlib.pyplot as plt

    dt = 0.01
    run_time = 1
    n_steps = int(run_time/dt)

    dmp = DMP(y0=1, g=-1, dt=dt, run_time=run_time)

    taus = [1/4, 1/2, 1, 2, 4]
    t = np.linspace(0, run_time, n_steps)

    for tau in taus:
        y, dy, ddy = dmp.rollout(tau=tau)
        print(y)
        plt.plot(t, y, label=f"tau={tau}")

    plt.xlabel("t")
    plt.ylabel("dmp(t)")
    plt.legend()
    plt.show()

def imitate_test():

    import matplotlib.pyplot as plt

    dmp = DMP(n_bfs=1000)
    y_des = np.sqrt(np.linspace(0, 3 * np.pi, 100) * 4.5)
    print(y_des.shape)

    if y_des.ndim == 1:
        y_des = y_des.reshape(1, len(y_des))
    dmp.imitate_path(y_des)
    trajectory, _, _ = dmp.rollout()
    print(dmp.w)

    for dim in range(y_des.shape[0]):

        plt.subplot(1, y_des.shape[0], dim+1)
        plt.plot(y_des[dim], label=f"desired")
        plt.plot(trajectory.T[dim], label=f"approximated")

    plt.legend()
    plt.show()



if __name__ == "__main__":
    imitate_test()
