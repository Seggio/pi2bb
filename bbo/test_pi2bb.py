from dmps.dmp_discrete import DMPsDiscrete
from executors.box2dsimexecutor import Box2DSimulatorExecutor
from bbo.pi2bb import Pi2BB

def test():

    n_dmps = 5

    def reward_function(x):
        if x["TOUCH_SENSORS"][6] != 0 and x["TOUCH_SENSORS"][7] != 0:
            return 1
        return 0

    dmp = DMPsDiscrete(n_dmps=n_dmps)
    executor = Box2DSimulatorExecutor(dmp, reward_function)
    bbo = Pi2BB(dmp, executor)

    bbo.train_loop()


if __name__ == "__main__":
    test()