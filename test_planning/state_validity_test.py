import os
import time

import pybullet as p
if os.environ.get('ROS_DISTRO'):
    import rospy
import timeit

from cairo_simulator.core.context import SawyerSimContext

from cairo_planning.collisions import DisabledCollisionsContext


def main():
    sim_context = SawyerSimContext()
    sim = sim_context.get_sim_instance()
    _ = sim_context.get_logger()
    scs = sim_context.get_state_space()
    svc = sim_context.get_state_validity()
    sawyer_robot = sim_context.get_robot()
    _ = sawyer_robot.get_simulator_id()
    _ = sim_context.get_sim_objects(['Ground'])[0]

    sawyer_robot.move_to_joint_pos([0, 0, 0, 0, 0, 0, 0])
    time.sleep(1)

    n_samples = 1000
    valid_samples = []
    starttime = timeit.default_timer()

    # Disabled collisions during planning with certain eclusions in place.
    with DisabledCollisionsContext(sim, [], []):
        print("Sampling start time is :", starttime)
        for i in range(0, n_samples):
            sample = scs.sample()
            if svc.validate(sample):
                print(sample)
                valid_samples.append(sample)
        print("The time difference is :", timeit.default_timer() - starttime)
        print("{} valid of {}".format(len(valid_samples), n_samples))

    p.disconnect()


if __name__ == "__main__":
    main()
