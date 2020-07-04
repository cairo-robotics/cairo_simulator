import sys
import os
import time
from functools import partial

import numpy as np
import pybullet as p
if os.environ.get('ROS_DISTRO'):
    import rospy

from cairo_simulator.simulator import Simulator, SimObject
from cairo_simulator.manipulators import Sawyer
from cairo_simulator.log import Logger
from planning.collision import self_collision_test, DisabledCollisionsContext
from cairo_simulator.utils import ASSETS_PATH
from cairo_simulator.link import get_link_pairs, get_joint_info_by_name

from cairo_motion_planning.geometric.state_space import SawyerConfigurationSpace
from cairo_motion_planning.sampling.state_validity import StateValidityChecker
from cairo_motion_planning.local.interpolation import interpolate_5poly

def main():

    ########################################################
    # Create the Simulator and pass it a Logger (optional) #
    ########################################################
    logger = Logger()
    sim = Simulator(logger=logger, use_ros=False, use_gui=True, use_real_time=True) # Initialize the Simulator

    #####################################
    # Create a Robot, or two, or three. #
    #####################################
    sawyer_robot = Sawyer("sawyer0", [0, 0, .9], fixed_base=1)

    #############################################
    # Create sim environment objects and assets #
    #############################################
    ground_plane = SimObject("Ground", "plane.urdf", [0,0,0])
    sawyer_id = sawyer_robot.get_simulator_id()

    ############
    # PLANNING #
    ############

    # Certain links in Sawyer seem to be permentently in self collision. This is how to remove them by name when getting all link pairs to check for self collision.
    excluded_pairs = [(get_joint_info_by_name(sawyer_id, "right_l1_2").idx, get_joint_info_by_name(sawyer_id, "right_l0").idx), 
                      (get_joint_info_by_name(sawyer_id, "right_l1_2").idx, get_joint_info_by_name(sawyer_id, "head").idx)]
    link_pairs = get_link_pairs(sawyer_id, excluded_pairs=excluded_pairs)
    self_collision_fn = partial(self_collision_test, robot=sawyer_robot, link_pairs=link_pairs)

    # Create a statevaliditychecker
    svc = StateValidityChecker(self_collision_fn)
    # Use a State Space specific to the environment and robots. Intrinsically uses UniformSampler but a different sampling technique could be used.
    scs = SawyerConfigurationSpace()

    valid_samples = []
    # Exclude the ground plane and the pedestal feet from disabled collisions.
    excluded_bodies = [ground_plane.get_simulator_id()] # the ground plane
    pedestal_feet_idx = get_joint_info_by_name(sawyer_id, 'pedestal_feet').idx
    excluded_body_link_pairs = [(sawyer_id, pedestal_feet_idx)]  # The (sawyer_idx, pedestal_feet_idx) tuple the ecluded from disabled collisions.

    time.sleep(1)

    # Disabled collisions during planning with certain eclusions in place.
    with DisabledCollisionsContext(sim, excluded_bodies, excluded_body_link_pairs):
        while True:
            sample = scs.sample()
            if svc.validate(sample):
                valid_samples.append(sample)
            if len(valid_samples) >= 1:
                break
        

    # Generate local plan between two points and execute local plan.
    steps = 100
    move_time = 200
    start_pos = [0]*7
    print(np.array(sawyer_robot.get_current_joint_states()[0:7]))
    print(np.array(valid_samples[0]))
    qt, qdt, qddt = interpolate_5poly(np.array(start_pos[0:7]), np.array(valid_samples[0]), steps)
    traj = list(zip([move_time * n/steps for n in range(0, steps)], [list(q) for q in qt]))
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