import numpy as np


class CanonicalSystem:
    """
           A class representing a discrete Canonical System

           Attributes
           ----------
           ax : float
               dumping coefficient of the exponential function
           run_time : float
               duration of the system to use in a rollout
           dt: float
                integration interval
           x0: float
                starting state of the CS
           x: float
                current state of the CS
           x_track: np.array
                array representing a complete rollout of the CS

           Methods
           -------
           rollout(timesteps=None)
               Executes a rollout of the CS with a specific number of timesteps

           step()
               Executes a single CS step
           """

    def __init__(self, dt, ax=1.0, x0=1.0, run_time=1.0):

        self.ax = ax

        self.run_time = run_time
        self.dt = dt
        self.timesteps = int(self.run_time / self.dt)

        # Setting initial x, current x, x values after a rollout
        self.x0 = x0
        self.x = None
        self.x_track = None

        self.reset_state()

    def reset_state(self):
        """Resets the CS state"""

        self.x = self.x0

    def rollout(self, **kwargs):
        """Executes a rollout of the CS with a specific number of timesteps"""

        if 'tau' in kwargs:
            timesteps = int(self.timesteps / kwargs['tau'])
        else:
            timesteps = self.timesteps
        self.x_track = np.zeros(timesteps)

        self.reset_state()
        for t in range(timesteps):
            self.x_track[t] = self.x
            self.step(**kwargs)

        return self.x_track

    def step(self, tau=1.0):
        """Executing a single step computing the current state value with Euler integration"""
        self.x += (-self.ax * self.x) * tau * self.dt
        return self.x


def main():
    from matplotlib import pyplot as plt

    run_time = 4.0
    dt = 0.01
    steps = int(run_time/dt)

    cs = CanonicalSystem(dt, run_time=run_time)
    t = np.linspace(0, run_time, steps)
    x_track = cs.rollout()
    plt.xlabel("TIME")
    plt.ylabel("CS")
    tau = 1/2
    x_track2 = cs.rollout(tau=tau)[:steps]
    plt.plot(t, x_track, label="CS tau=1")
    plt.plot(t, x_track2, label="CS tau=1/2")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()

