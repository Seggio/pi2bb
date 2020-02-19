from abc import ABC, abstractmethod


class Simulator(ABC):

    def __init__(self, rfn, update_params):

        self.set_reward_fn(rfn)
        self.update_params = update_params
        self.reset()

    @abstractmethod
    def set_simulator(self):
        pass

    @abstractmethod
    def set_reward_fn(self, rfn):
        pass

    @abstractmethod
    def render(self, **kwargs):
        pass

    @abstractmethod
    def step(self, action):
        pass

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def execute_simulator(self):
        pass

    def set_update_param(self, update_params):
        self.update_params = update_params


