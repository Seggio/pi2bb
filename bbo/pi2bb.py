from datetime import datetime
from dmps.dmp_discrete import DMPsDiscrete
from simulator.simulator import Simulator
from simulator.box2dsim import Box2DSimulator
from simulator.pybulletsim import PybulletSim
from utils.utils import *
import os


class Pi2BB:

    """Class implementing the PI2BB optimization algorithm"""

    def __init__(self, reward_fn, , rollouts_per_step=20, **kwargs):

        # Variance for the perturbation of the DMP parameters
        self.init_variance = self.variance = kwargs.get("variance", 1)

        # Initializing current vector of parameters
        self.curr_theta = {}
        self.set_theta()

        # Pi2BB Hyperparameters
        self.rollouts_per_step = rollouts_per_step
        self.max_steps = kwargs.get("max_steps", 100)
        self.curr_cost = kwargs.get("initial_cost", 0)
        self.improvement = 0
        self.converged_cost = kwargs.get("converged_cost", np.inf)
        self.temperature = kwargs.get("temperature", 1)  # Temperature for the softmax probability of costs
        self.test_freq = kwargs.get("test_freq", None)
        self.cumulative_cost = kwargs.get("cumulative_cost", False)
        self.train_render = kwargs.get("train_render", False)
        self.test_render = kwargs.get("test_render", False)

        # List containing train costs for each train step
        self.train_costs = []

        # Setup log/weights files
        time_format = "%d-%m-%Y_%I-%M-%S_%p"
        curr_time = datetime.now().strftime(time_format)
        self.log = kwargs.get("log", False)
        self.save_w = kwargs.get("save_w", False)

        if self.log:
            if not os.path.exists('results/logs'):
                os.makedirs('results/logs')
            self.logfile = f"./results/logs/log_{curr_time}"
            log_to_file(self.logfile, self.__str__())

        self.w_dir = f"./results/weights/{curr_time}"

    def train_loop(self):
        """Pi2BB train loop"""

        num_iter = 0
        while self.curr_cost < self.converged_cost and num_iter < self.max_steps:

            # Perturbation of DMPs parameters and compute cost for each perturbationn
            epsilons = []
            costs = []

            for _ in range(self.rollouts_per_step):

                e_k_y0, e_k_g, e_k_w = self.generate_perturbations()
                epsilons.append((e_k_y0, e_k_g, e_k_w))

                self.dmp.y0 += e_k_y0
                self.dmp.g += e_k_g
                self.dmp.w += e_k_w

                trajectory = self.compute_trajectory()
                tot_cost = self.execute_sim(trajectory, render=self.train_render)
                costs.append(tot_cost)
                self.update_dmp()  # Resetting y0,g,w of the DMPs after a rollout

            probs = self.compute_update_probabilities(costs)

            d_y0 = np.sum(np.array([x * y[0] for (x, y) in zip(probs, epsilons)]), axis=0)
            d_g = np.sum(np.array([x * y[1] for (x, y) in zip(probs, epsilons)]), axis=0)
            d_w = np.sum(np.array([x * y[2] for (x, y) in zip(probs, epsilons)]), axis=0)

            self.set_theta(self.curr_theta["y0"] + d_y0,
                           self.curr_theta["g"] + d_g,
                           self.curr_theta["w"] + d_w)
            self.update_dmp()  # Set DMPs parameters with new y0,g,w

            # Executes rollout to compute the current cost
            cost = self.execute_sim(self.compute_trajectory(), render=self.test_render)
            # self.update_variance(self.curr_cost, cost)
            self.curr_cost = cost
            self.train_costs.append(cost)

            if self.test_freq:
                if num_iter % self.test_freq == 0:
                    print(f"iteration:{num_iter +1}\tcost:{self.curr_cost:.2f}")

            if self.log:
                log_to_file(self.logfile, f"iteration:{num_iter +1}\tcost:{self.curr_cost:.2f}")

            num_iter += 1

        print(f"End after {num_iter} iterations\tFinal cost: {self.curr_cost:.2f}")

        if self.save_w:
            self.save_weights()

    def compute_trajectory(self):
        y, _, _ = self.dmp.rollout()
        return y

    def execute_sim(self, y, render=False, mode="human"):

        self.reset_sim()

        costs = []
        for t in range(y.shape[0]):
            costs.append(self.sim.step(y[t])[1])
            if render:
                self.sim.render(mode=mode)

        return np.sum(np.cumsum(costs))

    def update_variance(self, prev_cost, curr_cost):
        self.improvement += curr_cost - prev_cost
        self.variance = np.exp(self.improvement / 100) * self.init_variance

    def create_dmp(self):
        return DMPsDiscrete(n_dmps=self.ndim, n_bfs=self.nbfs, dt=self.dt)

    def update_dmp(self):
        self.dmp.set_params(self.curr_theta["y0"].copy(), self.curr_theta["g"].copy(), self.curr_theta["w"].copy())

    def set_curr_cost(self, cost):
        self.curr_cost = cost

    def init_params(self, **kwargs):
        y0 = kwargs.get("y0", np.zeros(self.ndim))
        g = kwargs.get("g", np.zeros(self.ndim))
        w = kwargs.get("w", np.zeros((self.ndim, self.nbfs)))
        return y0, g, w

    def generate_perturbations(self):

        e_k_y0 = np.random.randn(self.ndim) * self.variance
        e_k_g = np.random.randn(self.ndim) * self.variance
        e_k_w = np.random.randn(self.ndim, self.nbfs) * self.variance

        return e_k_y0, e_k_g, e_k_w

    def set_theta(self, y0, g, w):
        self.curr_theta["y0"] = y0
        self.curr_theta["g"] = g
        self.curr_theta["w"] = w

    def init_sim(self, reward_fn):
        self.sim = Simulator(reward_fn)

    def reset_sim(self):
        self.sim.reset()

    def compute_update_probabilities(self, costs):
        c = np.array(costs)
        exp = np.exp((c - np.max(c)) / self.temperature)
        return exp / np.sum(exp)

    def save_weights(self):
        if not os.path.exists(self.w_dir):
            os.makedirs(self.w_dir)
        np.save(os.path.join(self.w_dir, "y0"), self.curr_theta["y0"])
        np.save(os.path.join(self.w_dir, "g"), self.curr_theta["g"])
        np.save(os.path.join(self.w_dir, "w"), self.curr_theta["w"])

    def test(self, render=False, mode="human"):
        self.reset_sim()
        return self.execute_sim(self.compute_trajectory(), render=render, mode=mode)

    def __str__(self):
        return f"PI2BB: num_basis={self.nbfs},n_dim={self.ndim},rollouts_per_step={self.rollouts_per_step}"


def main():

    ndim = 5
    pi2bb = Pi2BB(ndim,
                  lambda x: 1 if (x["TOUCH_SENSORS"][6] != 0 and x["TOUCH_SENSORS"][7] != 0) else 0,
                  rollouts_per_step=30,
                  test_freq=50,
                  train_render=True)
    pi2bb.train_loop()


if __name__ == "__main__":
    main()
