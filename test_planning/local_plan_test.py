import sys
import os
import time

import numpy as np
import pybullet as p
if os.environ.get('ROS_DISTRO'):
    import rospy

from cairo_simulator.core.context import SawyerSimContext

from cairo_planning.collisions import DisabledCollisionsContext
from cairo_planning.local.interpolation import interpolate_5poly


def main():

    sim_context = SawyerSimContext()
    sim = sim_context.get_sim_instance()
    _ = sim_context.get_logger()
    state_space = sim_context.get_state_space()
    svc = sim_context.get_state_validity()
    sawyer_robot = sim_context.get_robot()
    _ = sawyer_robot.get_simulator_id()
    _ = sim_context.get_sim_objects(['Ground'])[0]

    sawyer_robot.move_to_joint_pos([0, 0, 0, 0, 0, 0, 0])
    time.sleep(1)
    ############
    # PLANNING #
    ############
    valid_samples = []
    # Disabled collisions during planning with certain eclusions in place.
    with DisabledCollisionsContext(sim, [], []):
        while True:
            sample = state_space.sample()
            if svc.validate(sample):
                valid_samples.append(sample)
            if len(valid_samples) >= 1:
                break

    # Generate local plan between two points and execute local plan.
    steps = 100
    move_time = 5
    start_pos = [0]*7
    print(np.array(sawyer_robot.get_current_joint_states()[0:7]))
    print(np.array(valid_samples[0]))
    qt, qdt, qddt = interpolate_5poly(
        np.array(start_pos[0:7]), np.array(valid_samples[0]), steps)
    traj = list(
        zip([move_time * n/steps for n in range(0, steps)], [list(q) for q in qt]))
    print(traj)
    sawyer_robot.execute_trajectory(traj)
    # Loop until someone shuts us down
    try:
        while True:
            sim.step()
    except KeyboardInterrupt:
        p.disconnect()
        sys.exit(0)


if __name__ == "__main__":
    main()
