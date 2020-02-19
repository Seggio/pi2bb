from pi2bb import Pi2BB
from utils.utils import load_weights
import numpy as np


def make_pi2bb(ndim=5, rollouts_per_step=30, max_steps=50, test_freq=None, nbfs=25, variance= 1, param=None):
    return Pi2BB(ndim,
                 lambda x: 1 if (x["TOUCH_SENSORS"][6] != 0 and x["TOUCH_SENSORS"][7] != 0) else 0,
                 rollouts_per_step=rollouts_per_step,
                 max_steps=max_steps,
                 test_freq=test_freq,
                 nbfs=nbfs,
                 variance=variance,
                 update_parameters={"update_type": "position",
                                    "body_name": "Object",
                                    "p_dict": {"x": param[0], "y": param[1]}}
                 )


def train_test():

    # Defining parameter type and values
    x_train = np.linspace(14, 16, 3)
    y_train = np.linspace(10, 15, 3)

    train_params = np.array(np.meshgrid(x_train, y_train)).T.reshape(-1, 2)

    ndmps = 5
    nbfs = 25
    rollouts_per_step = 30
    max_steps = 50
    test_freq = None
    variance = 1

    bb_algorithm = None
    w_dirs = []

    for q in train_params:

        it = 1
        cost = 0

        while cost < 1800:
            print(f"Started iteration {it} with q={q}")

            bb_algorithm = make_pi2bb(ndmps, rollouts_per_step, max_steps, test_freq, nbfs, variance, q)
            bb_algorithm.train_loop()

            trajectory = bb_algorithm.compute_trajectory()
            cost = bb_algorithm.execute_sim(trajectory)
            it += 1

        print(f"End after {it} trials, final reward = {cost}")
        bb_algorithm.test(render=True)
        bb_algorithm.save_weights()

        w_dirs.append((bb_algorithm.w_dir, cost))

    for ((directory,cost),q) in zip(w_dirs,train_params):

        bb_algorithm = make_pi2bb(param=q)

        y0, g, w = load_weights(directory)
        bb_algorithm.dmp.set_params(y0, g, w)

        print(f"Train cost: {cost}. Cost with loaded params: {bb_algorithm.test(render=True)}")


if __name__ == "__main__":
    train_test()
