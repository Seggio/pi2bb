import numpy as np
from abc import ABC, abstractmethod
from dmps.cs import CanonicalSystem


class DMPs(ABC):
    """
       An abstract class used to represent a DMP, implemented by discrete_dmps and parametric_dmps

       Attributes
       ----------
       n_dmps : int
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
            Position/Velocity/Acceleration after a rollout

       Methods
       -------
       rollout(timesteps=None)
           Executes a rollout of the DMPs with a specific number of timesteps

       step()
           Executes a single DMP step
       """

    def __init__(self, n_dmps=1, n_bfs=25, dt=.01, w=None,
                 y0=0, goal=1, ay=None, by=None, **kwargs):

        self.n_dmps = n_dmps
        self.n_bfs = n_bfs
        self.dt = dt

        if isinstance(y0, (int, float)):
            y0 = np.ones(self.n_dmps)*y0
        self.y0 = y0

        if isinstance(goal, (int, float)):
            goal = np.ones(self.n_dmps)*goal
        self.g = goal

        self.w = None
        self.initialize_weights(w)

        # Default value for ay/by to have the system critically dumped
        self.ay = np.ones(n_dmps) * 25. if ay is None else ay
        self.by = self.ay / 4. if by is None else by

        # set up the CS
        self.cs = CanonicalSystem(dt=self.dt, **kwargs)
        self.timesteps = self.cs.timesteps

        # set up the initial DMP state
        self.y = None
        self.dy = None
        self.ddy = None
        self.reset_state()

    def gen_front_term(self, x, dmp_num):
        """Generates the term x*(goal - y0) for a specific DoF """
        return x * (self.g[dmp_num] - self.y0[dmp_num])

    @abstractmethod
    def gen_psi(self, x):
        """Generates Gaussian activations for a specific x value """
        pass

    @abstractmethod
    def gen_centers(self):
        """Generates the Gaussians' ceters """
        pass

    @abstractmethod
    def gen_forcing_term(self, x, dmp_num, **kwargs):
        """Generate forcing term for a specific DoF and x value"""
        pass

    @abstractmethod
    def initialize_weights(self, w):
        """Initialize the DMPs' weights"""
        pass

    def rollout(self, timesteps=None, **kwargs):
        """Executes a complete rollout of the DMPs

        Retruns:
        y_track: trajectory array
        dy_track: velocitis array
        ddy_track: accelerations array"""

        self.reset_state()

        if timesteps is None:
            if 'tau' in kwargs:
                timesteps = int(self.timesteps / kwargs['tau'])
            else:
                timesteps = self.timesteps

        y_track = np.zeros((timesteps, self.n_dmps))
        dy_track = np.zeros((timesteps, self.n_dmps))
        ddy_track = np.zeros((timesteps, self.n_dmps))

        for t in range(timesteps):
            y_track[t], dy_track[t], ddy_track[t] = self.step(**kwargs)

        return y_track, dy_track, ddy_track

    def step(self, tau=1.0, **kwargs):
        """Executes a single DMP step"""

        # single step of the canonical system
        x = self.cs.step(tau=tau)

        # step for each DoF
        for d in range(self.n_dmps):

            # generate the forcing term
            forcing_term = kwargs.get("forcing_term", True)
            f = 0
            if forcing_term:
                f = self.gen_forcing_term(x, d, **kwargs)

            # Current DMP acceleration, velocity, position
            self.ddy[d] = (self.ay[d] *
                           (self.by[d] * (self.g[d] - self.y[d]) -
                           self.dy[d]) + f)

            self.dy[d] += self.ddy[d] * tau * self.dt
            self.y[d] += self.dy[d] * tau * self.dt

        return self.y, self.dy, self.ddy

    def reset_state(self):
        self.y = self.y0.copy()
        self.dy = np.zeros(self.n_dmps)
        self.ddy = np.zeros(self.n_dmps)
        self.cs.reset_state()

    def check_offset(self):
        """Checks the offsest g-y0, if 0 adds a noise to g in order to have a forcing term != 0"""
        for d in range(self.n_dmps):
            if self.y0[d] == self.g[d]:
                self.g[d] += 1e-4

    def get_params(self):

        return {"y0": self.y0,
                "g": self.g,
                "w": self.w}
