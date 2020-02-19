from realcomp.envs import REALCompEnv
from simulator.simulator import Simulator


class PybulletSim(Simulator):

    def __init__(self, rfn, update_params):
        self.simulator = None
        self.set_simulator()
        super().__init__(rfn, update_params)

    def set_simulator(self):
        self.simulator = REALCompEnv()
        self.simulator.robot.used_objects = ["table", "orange"]
        #self.simulator.render("human")

    def set_reward_fn(self, rfn):
        self.simulator.reward_func = rfn

    def render(self, **kwargs):
        self.simulator.render()

    def step(self, action):
        return self.simulator.step(action)

    def reset(self):
        self.simulator.reset()