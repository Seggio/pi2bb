from datetime import datetime
from utils.utils import *
import os


class Pi2BB:

    """Class implementing the PI2BB optimization algorithm"""

    def __init__(self, bb_fn, bb_fn_executor, **kwargs):

        # Black box function to optimize
        self.bb_fn = bb_fn

        # Initializing the bb_fn executor
        self.bb_fn_executor = bb_fn_executor

        # Initializing dictionary of parameters {"param_name": param_value}
        self.theta = {}
        self.init_theta()

        # Pi2BB Hyperparameters
        self.rollouts_per_step = kwargs.get("rollouts_per_step", 20)
        self.max_steps = kwargs.get("max_steps", 10000)
        self.curr_cost = kwargs.get("initial_cost", 0)
        self.improvement = 0
        self.converged_cost = kwargs.get("converged_cost", np.inf)
        # Temperature for the softmax probability of costs
        self.temperature = kwargs.get("temperature", 1)
        # Variance for the perturbation of the DMP parameters
        self.init_variance = self.variance = kwargs.get("variance", 1)

        self.test_freq = kwargs.get("test_freq", 0)

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

            epsilons = {param_name: [] for param_name in self.theta.keys()}
            costs = []

            for _ in range(self.rollouts_per_step):

                perts = self.generate_perturbations()

                for param_name in perts.keys():
                    epsilons[param_name].append(perts[param_name])
                    self.bb_fn.param_name += perts[param_name]

                tot_cost = self.bb_fn_executor.execute()

                costs.append(tot_cost)
                self.set_bb_fn_params()  # Resetting bb_fn parameters after a rollout

            update_probabilities = self.compute_update_probabilities(costs)

            for param_name in self.theta.keys():
                update = np.sum(np.array(epsilons[param_name]) * np.array(update_probabilities), axis=0)
                self.theta[param_name] += update

            self.set_bb_fn_params()  # Sets new bb_fn parameters

            # Executes rollout to compute the current cost
            cost = self.bb_fn_executor.execute(test=True)

            # self.update_variance(self.curr_cost, cost)
            self.curr_cost = cost
            self.train_costs.append(cost)

            if self.test_freq != 0 and num_iter % self.test_freq == 0:
                print(f"iteration:{num_iter +1}\tcost:{self.curr_cost:.2f}")

            if self.log:
                log_to_file(self.logfile, f"iteration:{num_iter +1}\tcost:{self.curr_cost:.2f}")

            num_iter += 1

        print(f"End after {num_iter} iterations\tFinal cost: {self.curr_cost:.2f}")

        if self.save_w:
            self.save_weights()

    def update_variance(self, prev_cost, curr_cost):
        self.improvement += curr_cost - prev_cost
        self.variance = np.exp(self.improvement / 100) * self.init_variance

    def set_curr_cost(self, cost):
        self.curr_cost = cost

    def compute_update_probabilities(self, costs):
        c = np.array(costs)
        exp = np.exp((c - np.max(c)) / self.temperature)
        return exp / np.sum(exp)

    def save_weights(self):
        if not os.path.exists(self.w_dir):
            os.makedirs(self.w_dir)
        for param_name, param in self.theta.items():
            np.save(os.path.join(self.w_dir, param_name), param)

    def init_theta(self):
        """Initialize a dictionary containing the black box function parameters"""
        for param_name, param_value in self.bb_fn.get_params.items():
            self.theta[param_name] = param_value.copy()

    def generate_perturbations(self):

        pert_dict = {}

        for param_name, param_value in self.theta.items():
            pert_dict[param_name] = np.random.randn(param_value.shape) * self.variance

        return pert_dict

    def set_bb_fn_params(self):

        for param_name in self.theta.keys():
            self.bb_fn.param_name = self.theta[param_name].copy()

    def __str__(self):
        return f"PI2BB: num_basis={self.nbfs},n_dim={self.ndim},rollouts_per_step={self.rollouts_per_step}"
