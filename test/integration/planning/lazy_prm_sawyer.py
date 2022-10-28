import os
import sys
from functools import partial
import time

import pybullet as p
if os.environ.get('ROS_DISTRO'):
    import rospy
import numpy as np

from cairo_simulator.core.sim_context import SawyerBiasedSimContext

from cairo_planning.collisions import DisabledCollisionsContext
from cairo_planning.local.interpolation import parametric_lerp
from cairo_planning.local.curve import JointTrajectoryCurve
from cairo_planning.planners import LazyPRM


def main():
    sim_context = SawyerBiasedSimContext()
    sim = sim_context.get_sim_instance()
    logger = sim_context.get_logger()
    state_space = sim_context.get_state_space()
    svc = sim_context.get_state_validity()
    sawyer_robot = sim_context.get_robot()
    _ = sawyer_robot.get_simulator_id()
    _ = sim_context.get_sim_objects(['Ground'])[0]

    sawyer_robot.move_to_joint_pos([0, 0, 0, 0, 0, 0, 0])
    time.sleep(1)
    
    start = [0, 0, 0, 0, 0, 0, 0] 
    end = [-1.308358671699429, 0.6967113296390992, -1.2503978032875767, -0.6960306042779365, -0.492062438070457, -0.6745904098030175, 2.8850870012999232]
    
    sawyer_robot.set_joint_state(start)
    with DisabledCollisionsContext(sim, [], []):
        #######
        # LazyPRM #
        #######
        # Use parametric linear interpolation with 10 steps between points.
        interp = partial(parametric_lerp, steps=10)
        # See params for PRM specific parameters
        prm = LazyPRM(state_space, svc, interp, params={
                  'n_samples': 4000, 'k': 8, 'ball_radius': 2.5})
        logger.info("Planning....")
        plan = prm.plan(np.array(start), np.array(end))
        # get_path() reuses the interp function to get the path between vertices of a successful plan
        path = prm.get_path(plan)
    if len(path) == 0:
        logger.info("Planning failed....")
        sys.exit(1)
    logger.info("Plan found....")

    # splinging uses numpy so needs to be converted
    path = [np.array(p) for p in path]
    # Create a MinJerk spline trajectory using JointTrajectoryCurve and execute
    jtc = JointTrajectoryCurve()
    traj = jtc.generate_trajectory(path, move_time=2)
    key = input("Press any key to excute plan.")

    try:
        prior_time = 0
        for i, point in enumerate(traj):
            if not svc.validate(point[1]):
                print("Invalid point: {}".format(point[1]))
                continue
            sawyer_robot.set_joint_state(point[1])
            time.sleep(point[0] - prior_time)
            prior_time = point[0]
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
