from pi2bb import Pi2BB
import os
import numpy as np
import matplotlib.pyplot as plt


def save_cost_improvement_logs_weights(nrollouts, nbasis, ntrials, **kwargs):

    reward_fn = kwargs.get("reward_fn",
                           lambda x: -1 if (x["TOUCH_SENSORS"][6] != 0 and x["TOUCH_SENSORS"][7] != 0) else 0)
    # Executng ntrials replicas of pi2bb for each combination nr,nb in nrollouts,nbasis
    for nr in nrollouts:
        for nb in nbasis:
            for t in range(ntrials):
                pi2bb = Pi2BB(5, reward_fn, rollouts_per_step=nr, nbfs=nb, test_render=False, max_steps=500)
                pi2bb.train_loop()
                print(f"End of replica {t + 1}")
            print("=" * 50)


# Plotting average results: fixed nbfs vs variable nrollouts, fixed nr vs variable nbfs
def plot_results_from_logs():

    results = {} # Dictionary of {(nrollouts,nbasis): [costs_over_time]}

    for file in os.listdir("results/logs"):
        path_to_log = os.path.join("results/logs", file)
        with open(path_to_log,"r") as f:
            nb, nr = tuple([x.split(":")[1] for x in (f.readline().rstrip().split("\t"))][1:])
            if (nb, nr) not in results:
                results[(nb, nr)] = []
            res = [float(x.rstrip().split(":")[-1]) for x in (f.readlines())]
            results[(nb, nr)].append(res)

    average_results = {(int(k[0]), int(k[1])): np.mean(v,axis=0) for k, v in results.items()}

    nbfs = sorted(set([x[0] for x in average_results.keys()]))
    nrollouts = sorted(set([x[1] for x in average_results.keys()]))

    for nb in nbfs:
        legends = []
        for nr in nrollouts:
            plt.plot([-x for x in average_results[(nb, nr)]])
            legends.append(f"nr:{nr}")
        plt.legend(legends)
        plt.xlabel("Iteration")
        plt.ylabel("Average Reward")
        plt.title(f"Fixed nb:{nb}")
        plt.show()

    for nr in nrollouts:
        legends = []
        for nb in nbfs:
            plt.plot([-x for x in average_results[(nb, nr)]])
            legends.append(f"nb:{nb}")
        plt.legend(legends)
        plt.xlabel("Iteration")
        plt.ylabel("Average Reward")
        plt.title(f"Fixed nr:{nr}")
        plt.show()


def main():
    # save_cost_improvement_logs_weights([10,20,30,50,100],[5,10,20,50,100],5)
    plot_results_from_logs()


if __name__ == "__main__":
    main()
