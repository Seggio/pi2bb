import numpy as np
from pi2bb import Pi2BB
from utils.utils import load_weights
from LWR import LWR
import scipy.spatial.distance as distance
import matplotlib.pyplot as plt

def main():

    g_regressors = []
    y0_regressors = []
    w_regressors = []

    g_dataset = []
    y0_dataset = []
    w_dataset = []

    def predict(q):
        return np.array([lwr.predict(q) for lwr in g_regressors]), \
               np.array([lwr.predict(q) for lwr in y0_regressors]), \
               np.array([[dim_lwr.predict(q) for dim_lwr in dmp_lwr] for dmp_lwr in w_regressors])


    def make_pi2bb(ndim=5, rollouts_per_step=30, max_steps=50, test_freq=None, nbfs=25, variance=1, param=None):
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

    x_train = np.linspace(14, 16, 3)
    y_train = np.linspace(10, 15, 3)

    train_params = np.array(np.meshgrid(x_train, y_train)).T.reshape(-1, 2)
    wdirs = ["./results/weights/" + wdir for wdir in ["wdir1", "wdir2", "wdir3", "wdir4",
                                                      "wdir5","wdir6","wdir7","wdir8","wdir9"]]

    for idx, q in enumerate(train_params):

        y0, g, w = load_weights(wdirs[idx])
        y0_dataset.append(y0)
        g_dataset.append(g)
        w_dataset.append(w)

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
    for d in range(5):

        dmp_regressors = []

        for dim in range(w_dataset.shape[-1]):
            ws = w_dataset[:, d, dim]
            dmp_regressors.append(LWR(train_params, ws))

        w_regressors.append(dmp_regressors)

    for id, q in enumerate(train_params):

        g, y0, w = predict(q)
        pi2bb = make_pi2bb(param=q)
        pi2bb.dmp.set_params(y0,g,w)

        cost = pi2bb.test()
        print(f"q={q}, cost={cost}")
        print(f"y0 distance={distance.sqeuclidean(y0, y0_dataset[id])}\ng distance={distance.sqeuclidean(g,g_dataset[id])}")
        print(f"Weights distance: {np.sqrt(np.sum((w_dataset[id]-w)**2))}")


    #Testing the generalization

    x_test = np.linspace(14, 16, 40)
    y_test = np.linspace(10, 15, 40)

    test_params = np.array(np.meshgrid(x_test, y_test)).T.reshape(-1, 2)
    costs = []
    for q in test_params:
        g, y0, w = predict(q)
        pi2bb = make_pi2bb(param=q)
        pi2bb.dmp.set_params(y0,g,w)

        cost = pi2bb.test()
        costs.append(cost)

    costs = np.array(costs).reshape((40,40))
    plt.subplot(111, aspect="auto")
    plt.imshow(costs, cmap='hot', interpolation='nearest')
    plt.colorbar()
    plt.show()


if __name__ == "__main__":
    main()
