import os
import sys
from functools import partial

import pybullet as p
if os.environ.get('ROS_DISTRO'):
    import rospy
import numpy as np

from cairo_simulator.core.context import SawyerSimContext
from cairo_simulator.core.link import get_joint_info_by_name

from cairo_planning.collisions import DisabledCollisionsContext
from cairo_planning.local.interpolation import parametric_lerp
from cairo_planning.local.curve import JointTrajectoryCurve
from cairo_planning.planners import PRM


def main():
    sim_context = SawyerSimContext()
    sim = sim_context.get_sim_instance()
    logger = sim_context.get_logger()
    state_space = sim_context.get_state_space()
    svc = sim_context.get_state_validity()
    sawyer_robot = sim_context.get_robot()
    sawyer_id = sawyer_robot.get_simulator_id()
    ground_plane = sim_context.get_sim_objects(['Ground'])[0]

    # Exclude the ground plane and the pedestal feet from disabled collisions.
    excluded_bodies = [ground_plane.get_simulator_id()]  # the ground plane
    pedestal_feet_idx = get_joint_info_by_name(sawyer_id, 'pedestal_feet').idx
    # The (sawyer_idx, pedestal_feet_idx) tuple the ecluded from disabled collisions.
    excluded_body_link_pairs = [(sawyer_id, pedestal_feet_idx)]

    ############
    # PLANNING #
    ############
    # Disabled collisions during planning with certain eclusions in place.
    with DisabledCollisionsContext(sim, excluded_bodies, excluded_body_link_pairs):
        #######
        # PRM #
        #######
        # Use parametric linear interpolation with 10 steps between points.
        interp = partial(parametric_lerp, steps=10)
        # See params for PRM specific parameters
        prm = PRM(state_space, svc, interp, params={
                  'n_samples': 3000, 'k': 6, 'ball_radius': 2.0})
        logger.info("Planning....")
        plan = prm.plan(np.array([0, 0, 0, 0, 0, 0, 0]), np.array([1.5262755737449423, -0.1698540226273928,
                                                                   2.7788151824762055, 2.4546623466066135, 0.7146948867821279, 2.7671787952787184, 2.606128412644311]))
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
