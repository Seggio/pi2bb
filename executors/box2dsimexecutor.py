from CNR_170608_SOURCE_box2d_simulation.box2dsim.envs.Box2DSim_env import Box2DSimOneArmEnv
from CNR_170608_SOURCE_box2d_simulation.box2dsim.envs import JsonToPyBox2D
import numpy as np


class Box2DSimulatorExecutor:

    def __init__(self, dmp, rfn, update_params=None):

        self.simulator = Box2DSimOneArmEnv()
        self.simulator.set_reward_fun(rfn)
        self.update_params = update_params
        self.dmp = dmp
        self.reset_simulator()

    def update_body_vertices(self, v_dict, body_name="Object"):
        self.simulator.sim.bodies[body_name].fixtures[0].shape.vertices = JsonToPyBox2D.rubeVecArrToB2Vec2Arr(v_dict)

    def update_body_position(self, p_dict, body_name="Object"):
        self.simulator.sim.bodies[body_name].position = JsonToPyBox2D.rubeVecToB2Vec2(p_dict)

    def reset_simulator(self):

        self.simulator.reset()

        if self.update_params:
            if self.update_params["update_type"] == "vertices":
                self.update_body_vertices(self.update_params["v_dict"], self.update_params["body_name"])
            elif self.update_params["update_type"] == "position":
                self.update_body_position(self.update_params["p_dict"], self.update_params["body_name"])
            else:
                raise ValueError("Invalid update params")

    def set_update_params(self, update_params):
        self.update_params = update_params

    def execute(self, test=False):

        self.reset_simulator()
        trajectory, _, _ = self.dmp.rollout()

        costs = []

        for t in range(trajectory.shape[0]):
            costs.append(self.simulator.step(trajectory[t])[1])
        if test:
            self.simulator.render()

        return np.sum(np.cumsum(costs))
