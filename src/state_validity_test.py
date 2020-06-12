import os

import pybullet as p
if os.environ.get('ROS_DISTRO'):
    import rospy

from cairo_simulator.simulator import Simulator, SimObject
from cairo_simulator.manipulators import Sawyer
from cairo_simulator.log import Logger
from planning.collision import self_collision_test, DisabledCollisionsContext
from cairo_simulator.utils import ASSETS_PATH
from cairo_simulator.link import get_link_pairs, get_joint_info_by_name

from cairo_motion_planning.samplers import UniformSampler
from cairo_motion_planning.state_space import SawyerConfigurationSpace
from cairo_motion_planning.state_validity import StateValidyChecker


from functools import partial

import timeit

def main():
    ################################
    # Environment Checks and Flags #
    ################################
    if os.environ.get('ROS_DISTRO'):
        rospy.init_node("CAIRO_Sawyer_Simulator")
        use_ros = True
    else:
        use_ros = False

    ########################################################
    # Create the Simulator and pass it a Logger (optional) #
    ########################################################
    logger = Logger()
    sim = Simulator(logger=logger, use_ros=use_ros, use_gui=True, use_real_time=False) # Initialize the Simulator

    #####################################
    # Create a Robot, or two, or three. #
    #####################################
    sawyer_robot = Sawyer("sawyer0", [0, 0, 0.9])

    #############################################
    # Create sim environment objects and assets #
    #############################################
    ground_plane = SimObject("Ground", "plane.urdf", [0,0,0])
    table = SimObject("Table", ASSETS_PATH + 'table.sdf', (0.9, 0, 0), (0, 0, 1.5708)) # Table rotated 90deg along z-axis
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
    svc = StateValidyChecker(self_collision_fn)
    # Use a State Space specific to the environment and robots.
    scs = SawyerConfigurationSpace()
    # Create a sampling technique.
    sampler = UniformSampler(scs.get_bounds())

    n_samples = 1000
    valid_samples = []
    starttime = timeit.default_timer()

    # Exclude the ground plane and the pedestal feet from disabled collisions.
    excluded_bodies = [ground_plane.get_simulator_id()] # the ground plane
    pedestal_feet_idx = get_joint_info_by_name(sawyer_id, 'pedestal_feet').idx
    excluded_body_link_pairs = [(sawyer_id, pedestal_feet_idx)]  # The (sawyer_idx, pedestal_feet_idx) tuple the ecluded from disabled collisions.

    # Disabled collisions during planning with certain eclusions in place.
    with DisabledCollisionsContext(sim, excluded_bodies, excluded_body_link_pairs):
        print("Sampling start time is :",starttime)
        for i in range(0, n_samples):
            sample = sampler.sample()
            if svc.validate(sample):
                valid_samples.append(sample)
        print("The time difference is :", timeit.default_timer() - starttime)
        print("{} valid of {}".format(len(valid_samples), n_samples))
    
    p.disconnect()


if __name__ == "__main__":
    main()