import os
import sys
from functools import partial

import pybullet as p
if os.environ.get('ROS_DISTRO'):
    import rospy
import numpy as np 

from cairo_simulator.core.simulator import Simulator, SimObject
from cairo_simulator.devices.manipulators import Sawyer
from cairo_simulator.core.log import Logger
from cairo_simulator.core.utils import ASSETS_PATH
from cairo_simulator.core.link import get_link_pairs, get_joint_info_by_name

from cairo_planning.collisions import self_collision_test, DisabledCollisionsContext
from cairo_planning.geometric.state_space import SawyerConfigurationSpace
from cairo_planning.sampling import StateValidityChecker
from cairo_planning.local.interpolation import parametric_lerp
from cairo_planning.trajectory.curve import JointTrajectoryCurve
from cairo_planning.planners import PRM

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
    sim = Simulator(logger=logger, use_ros=use_ros, use_gui=True, use_real_time=True) # Initialize the Simulator

    #####################################
    # Create a Robot, or two, or three. #
    #####################################
    sawyer_robot = Sawyer("sawyer0", [0, 0, 0.9], fixed_base=1)

    #############################################
    # Create sim environment objects and assets #
    #############################################
    ground_plane = SimObject("Ground", "plane.urdf", [0,0,0])
    sawyer_id = sawyer_robot.get_simulator_id()

    # Exclude the ground plane and the pedestal feet from disabled collisions.
    excluded_bodies = [ground_plane.get_simulator_id()] # the ground plane
    pedestal_feet_idx = get_joint_info_by_name(sawyer_id, 'pedestal_feet').idx
    excluded_body_link_pairs = [(sawyer_id, pedestal_feet_idx)]  # The (sawyer_idx, pedestal_feet_idx) tuple the ecluded from disabled collisions.

    ############
    # PLANNING #
    ############
    # Disabled collisions during planning with certain eclusions in place.
    with DisabledCollisionsContext(sim, excluded_bodies, excluded_body_link_pairs):
        #########################
        # STATE SPACE SELECTION #
        #########################
        # This inherently uses UniformSampler but a different sampling class could be injected.
        state_space = SawyerConfigurationSpace()
        ##############################
        # STATE VALIDITY FORMULATION #
        ##############################
        # Certain links in Sawyer seem to be permentently in self collision. This is how to remove them by name when getting all link pairs to check for self collision.
        excluded_pairs = [(get_joint_info_by_name(sawyer_id, "right_l1_2").idx, get_joint_info_by_name(sawyer_id, "right_l0").idx), 
                        (get_joint_info_by_name(sawyer_id, "right_l1_2").idx, get_joint_info_by_name(sawyer_id, "head").idx)]
        link_pairs = get_link_pairs(sawyer_id, excluded_pairs=excluded_pairs)
        self_collision_fn = partial(self_collision_test, robot=sawyer_robot, link_pairs=link_pairs)
        # In this case, we only have a self_col_fn.
        svc = StateValidityChecker(self_col_func=self_collision_fn, col_func=None, validity_funcs=None)
        #######
        # PRM #
        #######
        # Use parametric linear interpolation with 10 steps between points.
        interp = partial(parametric_lerp, steps=10)
        # See params for PRM specific parameters
        prm = PRM(state_space, svc, interp, params={'n_samples': 6000, 'k': 6, 'ball_radius': 2.0})
        logger.info("Planning....")
        plan = prm.plan(np.array([0, 0, 0, 0, 0, 0, 0]), np.array([1.5262755737449423, -0.1698540226273928, 2.7788151824762055, 2.4546623466066135, 0.7146948867821279, 2.7671787952787184, 2.606128412644311]))
        # get_path() reuses the interp function to get the path between vertices of a successful plan 
        path = prm.get_path(plan)
    if len(path) == 0:
            logger.info("Planning failed....")
            sys.exit(1)
    logger.info("Plan found....")
    print(len(path))
    ##########
    # SPLINE #
    ##########
    # splinging uses numpy so needs to be converted
    path = [np.array(p) for p in path]
    # Create a MinJerk spline trajectory using JointTrajectoryCurve and execute
    jtc = JointTrajectoryCurve()
    traj = jtc.generate_trajectory(path, move_time=10)
    sawyer_robot.execute_trajectory(traj)
    try:
        while True:
            sim.step()
    except KeyboardInterrupt:
        p.disconnect()
        sys.exit(0)

if __name__ == "__main__":
    main()