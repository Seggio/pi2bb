from pi2bb import Pi2BB
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.image as mgimg
import matplotlib
import numpy as np
import os
from _datetime import datetime
import gc


def make_pi2bb(ndim, rollouts_per_step, max_steps, test_freq, nbfs, variance, param):
    return Pi2BB(ndim,
                 lambda x: 1 if (x["TOUCH_SENSORS"][6] != 0 and x["TOUCH_SENSORS"][7] != 0) else 0,
                 rollouts_per_step=rollouts_per_step,
                 max_steps=max_steps,
                 test_freq=test_freq,
                 nbfs=nbfs,
                 variance=variance,
                 verbose_test=True,
                 log=True,
                 update_parameters= {"update_type": "position",
                                     "body_name": "Object",
                                     "p_dict": {"x": 15.0, "y": param}}
                 )


def test():

    matplotlib.use('TkAgg')

    # PI2BB hyperparameters
    ndmps = 5
    nbfs = 25
    rollouts_per_step = 30
    max_steps = 50
    test_freq = 10
    variance = 1

    train_q = [14, 15, 16]
    pi2bb = None

    for q in train_q:

        it = 1
        cost = 0

        while cost < 1500:

            print(f"Started iteration {it} with q={q}")

            pi2bb = make_pi2bb(ndmps, rollouts_per_step, max_steps, test_freq, nbfs, variance, q)

            pi2bb.train_loop()

            trajectory = pi2bb.compute_trajectory()
            cost = pi2bb.execute_sim(trajectory, render="True", mode="offline")
            it += 1

        costs = pi2bb.train_costs

        fig, (cost_plot, trajectory_anim) = plt.subplots(1, 2, figsize=(30, 15))
        fig.suptitle(f"Test with q={q}")
        cost_plot.set_title('Number of iterations vs Cost')
        cost_plot.set_yticks(list(np.arange(0, 2100, 100)))
        cost_plot.plot([i+1 for i in range(max_steps)], costs)
        trajectory_anim.set_title("Final Trajectory")
        trajectory_anim.axis("off")

        frames = []
        for i in range(pi2bb.dmp.timesteps):
            impath = "./frames/frame%06d.png" % i
            frames.append([trajectory_anim.imshow(mgimg.imread(impath))])

        anim = animation.ArtistAnimation(fig, frames, interval=30, blit=True, repeat=False)

        time_format = "%d-%m-%Y_%I-%M-%S_%p"
        curr_time = datetime.now().strftime(time_format)
        if not os.path.exists('./results/gifs'):
            os.makedirs('./results/gifs')
        anim.save(f'./results/gifs/result_50steps_{q}_{curr_time}.gif', writer=animation.PillowWriter(fps=30))

        # Clear the current axes.
        plt.cla()
        # Clear the current figure.
        plt.clf()
        # Closes all the figure windows.
        plt.close('all')
        gc.collect()


if __name__ == "__main__":
    test()
