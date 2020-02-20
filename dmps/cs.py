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

           Methods
           -------
<<<<<<< Updated upstream
           rollout(tau=1.)
               Executes a rollout of the CS with a specific frequency value (default 1)

           step(tau=1.)
               Executes a single CS step with a specific frequency value (default 1)
=======
           rollout(tau=1)
               Executes a rollout of the CS with a specific number of timesteps

           step(tau=1)
               Executes a single CS step
>>>>>>> Stashed changes
           """

    def __init__(self, dt, ax=1.0, x0=1.0, run_time=1.0):

        self.ax = ax

        self.run_time = run_time
        self.dt = dt
<<<<<<< Updated upstream
=======
        self.n_steps = int(self.run_time / self.dt)
>>>>>>> Stashed changes

        # Setting initial x, current x
        self.x0 = x0
        self.x = None
<<<<<<< Updated upstream
=======

        self.reset_state()
>>>>>>> Stashed changes

    def reset_state(self):
        """Resets the CS state"""

        self.x = self.x0

<<<<<<< Updated upstream
    def rollout(self, tau=1.):
        """Executes a rollout of the CS"""

        n_steps = int(self.run_time/self.dt)
        x_track = np.zeros(n_steps)

        self.reset_state()
        for t in range(n_steps):
            x_track[t] = self.x
            self.step(tau=tau)

        return x_track
=======
    def execute_rollout(self, tau=1):
        """Executes a rollout of the CS with a specific number of timesteps"""

        x_track = np.zeros(self.n_steps)

        self.reset_state()

        for t in range(self.n_steps):
            x_track[t] = self.x
            self.step(tau)
>>>>>>> Stashed changes

        return x_track

    def step(self, tau):
        """Executing a single step updating the current state with Euler integration"""
        self.x += (-self.ax * self.x) * tau * self.dt


def main():
    import matplotlib.pyplot as plt

    run_time = 4.0
    dt = 0.01
    steps = int(run_time / dt)

    cs = CanonicalSystem(dt, run_time=run_time)
    t = np.linspace(0, run_time, steps)

    plt.xlabel("t")
    plt.ylabel("cs(t)")
    taus = [1/4, 1/2, 1, 2]

    for tau in taus:
        x_track = cs.rollout(tau=tau)
        plt.plot(t, x_track, label=f"tau={tau}")

    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()

