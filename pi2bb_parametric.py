from pi2bb import Pi2BB
from dmps.dmp_parametric import DMPsParametric
import numpy as np


class Pi2BB_parametric(Pi2BB):

    """Class implementing the parametric PI2BB algorithm"""

    def __init__(self, ndim, reward_fn, minq, maxq, nbfsq, **kwargs):

        self.minq = minq
        self.maxq = maxq
        self.nbfsq = nbfsq
        self.q = kwargs.get("q", None)

        self.loess_g = []
        self.loess_y0 = []

        super().__init__(ndim, reward_fn, **kwargs)

    def create_dmp(self):
        return DMPsParametric(self.ndim, self.nbfs, self.minq, self.maxq, self.nbfsq, dt=self.dt)

    def init_params(self, **kwargs):

        y0 = kwargs.get("y0", np.random.randn(self.ndim) * self.variance)
        g = kwargs.get("g", np.random.randn(self.ndim) * self.variance)
        w = kwargs.get("w", np.random.randn(self.ndim, self.nbfs, self.nbfsq) * self.variance)
        return y0, g, w

    def generate_perturbations(self):

        e_k_y0 = np.random.randn(self.ndim) * self.variance
        e_k_g = np.random.randn(self.ndim) * self.variance
        e_k_w = np.random.randn(self.ndim, self.nbfs, self.nbfsq) * self.variance

        return e_k_y0,e_k_g, e_k_w

    def compute_trajectory(self):
        y, _, _ = self.dmp.rollout(q=self.q)
        return y

    def update_q(self, q):
        self.q = q

    def compute_vertices(self):
        return {"x": [self.q/2, self.q/2, -self.q/2, -self.q/2], "y": [self.q/2, - self.q/2, self.q/2, - self.q/2]}

    def compute_position(self):
        return {"x": 15.0, "y": self.q}

    def parametric_train_loop(self, q_vals, pert="position"):

        goals = []
        y0s = []

        for q in q_vals:

            self.update_q(q)
            print(f"Starting iteration with q={self.q}")

            assert pert == "position" or pert == "vertices"

            if pert == "position":
                update_params = {"update_type": "position",
                                 "body_name": "Object",
                                 "p_dict": self.compute_position()}
            else:
                update_params = {"update_type": "vertices",
                                 "body_name": "Object",
                                 "v_dict": self.compute_vertices()}

            self.sim.set_update_params(update_params)
            self.sim.reset()
            super().train_loop()

            goals.append(self.compute_trajectory()[-1])
            y0s.append(self.compute_trajectory()[-1])

        goals = np.array(goals)
        y0s = np.array(y0s)
        for dim in range(goals.shape[-1]):
            qs = np.array(q_vals)
            gs = goals[:, dim]
            ys = y0s[:, dim]

            self.loess_g.append(Loess(qs, gs))
            self.loess_y0.append(Loess(qs, ys))

    def predict_goal(self, q, window):
        return np.array([loess.estimate(q, window) for loess in self.loess_g]), \
               np.array([loess.estimate(q, window) for loess in self.loess_y0])

    def __str__(self):
        return f"PI2BB_parametric: num_basis={self.nbfs},num_basis_q={self.nbfsq}" \
               f",n_dim={self.ndim},rollouts_per_step={self.rollouts_per_step}"


def main():

    ndim = 5
    minq = 10
    maxq = 20
    nbfsq = 10

    pi2bb_parametric = Pi2BB_parametric(ndim,
                                        lambda x: -1 if (x["TOUCH_SENSORS"][6] != 0 and x["TOUCH_SENSORS"][7] != 0)
                                        else 0,
                                        minq,
                                        maxq,
                                        nbfsq,
                                        rollouts_per_step=200,
                                        test_freq=20,
                                        max_steps=200)

    train_q = [5, 10, 20, 15, 12, 8, 11, 12]
    test_q = [7, 18, 13, 14, 12.5, 14, 9, 6, 8]

    print(f"Training the system with q = {train_q}")
    print("="*30)
    pi2bb_parametric.parametric_train_loop(train_q)
    print("=" * 30)
    print(f"Testing the system with q ={test_q}")

    for qtest in test_q:

        pi2bb_parametric.update_q(qtest)

        goal, y0 = pi2bb_parametric.predict_goal(qtest, len(train_q))
        pi2bb_parametric.dmp.set_g(goal)
        pi2bb_parametric.dmp.set_y0(y0)

        cost = pi2bb_parametric.test(render=True)

        print(f"Cost for q = {qtest}: {cost}")
        print("=" * 30)


if __name__ == "__main__":
    main()

