from pi2bb import Pi2BB
from LWR import LWR
import numpy as np
import scipy.spatial.distance as distance


def main():

    # Defining parameter type and values
    x_train = np.linspace(14, 16, 3)
    y_train = np.linspace(10, 15, 3)

    x_test = np.linspace(14, 16, 25)
    y_test = np.linspace(10, 15, 25)

    train_params = np.array(np.meshgrid(x_train, y_train)).T.reshape(-1, 2)
    test_params = np.array(np.meshgrid(x_test, y_test)).T.reshape(-1, 2)

    pi2bb = None
    trajectory = None

    # PI2BB hyperparameters
    ndmps = 5
    nbfs = 25
    rollouts_per_step = 30
    max_steps = 100
    test_freq = None
    variance = 1

    # Initializing regressors
    g_regressors = []
    y0_regressors = []
    w_regressors = []

    g_dataset = []
    y0_dataset = []
    w_dataset = []

    train_costs = []
    test_tr_costs = []
    test_costs = []

    def predict(q):
        return np.array([lwr.predict(q) for lwr in g_regressors]), \
               np.array([lwr.predict(q) for lwr in y0_regressors]), \
               np.array([[dim_lwr.predict(q) for dim_lwr in dmp_lwr] for dmp_lwr in w_regressors])

    def make_pi2bb(ndim, rollouts_per_step, max_steps, test_freq, nbfs, variance, param):
        return Pi2BB(ndim,
                     lambda x: 1 if (x["TOUCH_SENSORS"][6] != 0 and x["TOUCH_SENSORS"][7] != 0) else 0,
                     rollouts_per_step=rollouts_per_step,
                     max_steps=max_steps,
                     test_freq=test_freq,
                     nbfs=nbfs,
                     variance=variance,
                     log=True,
                     update_parameters={"update_type": "position",
                                        "body_name": "Object",
                                        "p_dict": {"x": param[0], "y": param[1]}}
                     )

    for q in train_params:

        it = 1
        cost = 0

        while cost < 2000:

            print(f"Started iteration {it} with q={q}")

            pi2bb = make_pi2bb(ndmps, rollouts_per_step, max_steps, test_freq, nbfs, variance, q)
            pi2bb.train_loop()

            trajectory = pi2bb.compute_trajectory()
            cost = pi2bb.execute_sim(trajectory)
            it += 1

        g_dataset.append(trajectory[-1])
        y0_dataset.append(trajectory[0])
        w_dataset.append(pi2bb.dmp.w)
        train_costs.append((q, cost))

    g_dataset = np.array(g_dataset)
    y0_dataset = np.array(y0_dataset)
    w_dataset = np.array(w_dataset)

    # Creating goal and starting point regressors
    for dim in range(g_dataset.shape[-1]):
        gs = g_dataset[:, dim]
        y0s = y0_dataset[:, dim]

        g_regressors.append(LWR(train_params, gs))
        y0_regressors.append(LWR(train_params, y0s))

    # Creating weights regressors
    for d in range(ndmps):

        dmp_regressors = []

        for dim in range(w_dataset.shape[-1]):
            ws = w_dataset[:, d, dim]
            dmp_regressors.append(LWR(train_params, ws))

        w_regressors.append(dmp_regressors)

    # Testing the generalization
    for id, q in enumerate(train_params):

        g, y0, w = predict(q)
        pi2bb = make_pi2bb(ndmps,rollouts_per_step, max_steps, test_freq, nbfs, variance, q)
        pi2bb.dmp.set_g(g)
        pi2bb.dmp.set_y0(y0)
        pi2bb.dmp.set_w(w)

        cost = pi2bb.test(render=True)
        print(f"q={q}, cost={cost}")
        print(f"y0 distance={distance.sqeuclidean(y0, y0_dataset[id])}\ng distance={distance.sqeuclidean(g,g_dataset[id])}")
        print(f"Weights distance: {np.sqrt(np.sum((w_dataset[id]-w)**2))}")
        test_tr_costs.append((q,cost))

    for q in test_params:
        g, y0, w = predict(q)
        pi2bb = make_pi2bb(ndmps, rollouts_per_step, max_steps, test_freq, nbfs, variance, q)
        pi2bb.dmp.set_g(g)
        pi2bb.dmp.set_y0(y0)
        pi2bb.dmp.set_w(w)

        cost = pi2bb.test(render=True)
        print(f"q={q}, cost={cost}")
        test_costs.append((q,cost))

    with open("result_train.txt", "w") as f:
        f.write("TRAINING RESULTS\n")
        for q,cost in train_costs:
            f.write(f"{q} {cost}\n")
        f.write("TEST ON TRAIN POINTS\n")
        for q,cost in test_tr_costs:
            f.write(f"{q} {cost}\n")
        f.write("TEST RESULTS\n")
        for q,cost in test_costs:
            f.write(f"{q} {cost}\n")


if __name__ == "__main__":
    main()
