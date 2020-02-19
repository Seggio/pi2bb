from pi2bb import Pi2BB

def main():

    pi2bb = Pi2BB(9, lambda x: 1 if (x["touch_sensors"][0] != 0
                                     and x["touch_sensors"][1] != 0
                                     and x["touch_sensors"][2] != 0
                                     and x["touch_sensors"][3] != 0) else 0,
                  sim_type="pybullet",
                  max_steps=100,
                  test_render=True,
                  rollouts_per_step=50)
    pi2bb.train_loop()


if __name__ == "__main__":
    main()