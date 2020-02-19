from CNR_170608_SOURCE_box2d_simulation.box2dsim.envs import Box2DSimOneArmEnv, JsonToPyBox2D
from simulator.simulator import Simulator


class Box2DSimulator(Simulator):

    def __init__(self, rfn, update_params):
        self.simulator = None
        self.set_simulator()
        super().__init__(rfn, update_params)

    def set_simulator(self):
        self.simulator = Box2DSimOneArmEnv()

    def set_reward_fn(self, rfn):
        self.simulator.set_reward_fun(rfn)

    def update_body_vertices(self, v_dict, body_name="Object"):
        self.simulator.sim.bodies[body_name].fixtures[0].shape.vertices = JsonToPyBox2D.rubeVecArrToB2Vec2Arr(v_dict)

    def update_body_position(self, p_dict, body_name="Object"):
        self.simulator.sim.bodies[body_name].position = JsonToPyBox2D.rubeVecToB2Vec2(p_dict)

    def render(self, mode="human"):
        self.simulator.render(mode=mode)

    def step(self, action):
        return self.simulator.step(action)

    def reset(self):

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
