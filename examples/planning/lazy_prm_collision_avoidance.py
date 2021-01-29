import os
import sys
from functools import partial
import time

import pybullet as p
if os.environ.get('ROS_DISTRO'):
    import rospy
import numpy as np

from cairo_simulator.core.sim_context import SawyerSimContext

from cairo_simulator.core.utils import ASSETS_PATH
from cairo_planning.collisions import DisabledCollisionsContext
from cairo_planning.local.interpolation import parametric_lerp
from cairo_planning.local.curve import JointTrajectoryCurve
from cairo_planning.planners import LazyPRM


def main():
    config = {}
    config["sim_objects"] = [
        {
            "object_name": "Ground",
            "model_file_or_sim_id": "plane.urdf",
            "position": [0, 0, 0]
        },
        {
            "object_name": "Table",
            "model_file_or_sim_id": ASSETS_PATH + 'table.sdf',
            "position": [.6, -.8, 1.0],
            "orientation":  [0, 0, 1.5708],
            "fixed_base": 1
        }
    ]
    sim_context = SawyerSimContext(config)
    sim = sim_context.get_sim_instance()
    logger = sim_context.get_logger()
    state_space = sim_context.get_state_space()
    svc = sim_context.get_state_validity()
    sawyer_robot = sim_context.get_robot()
    _ = sawyer_robot.get_simulator_id()
    _ = sim_context.get_sim_objects(['Ground'])[0]


    start = [0, 0, 0, 0, 0, 0, 0]
   
    goal = [-1.9622245072067646, 0.8439858364277937, 1.3628459180018329, -0.2383928041974519, -2.7327884695211555, -2.2177502341009134, -0.08992133311928363]

    # sawyer_robot.move_to_joint_pos(goal)
    # time.sleep(5)
    sawyer_robot.move_to_joint_pos(start)
    time.sleep(5)
 
    with DisabledCollisionsContext(sim, [], []):
        #######
        # LazyPRM #
        #######
        # Use parametric linear interpolation with 10 steps between points.
        interp = partial(parametric_lerp, steps=10)
        # See params for PRM specific parameters
        prm = LazyPRM(state_space, svc, interp, params={
                  'n_samples': 8000, 'k': 30, 'planning_attempts': 5, 'ball_radius': 2.0})
        logger.info("Planning....")
        plan = prm.plan(np.array(start), np.array(goal))
        # get_path() reuses the interp function to get the path between vertices of a successful plan
        path = prm.get_path(plan)
    if len(path) == 0:
        logger.info("Planning failed....")
        sys.exit(1)
    logger.info("Plan found....")
    input("Press any key to continue...")
    # splining uses numpy so needs to be converted
    path = [np.array(p) for p in path]
    # Create a MinJerk spline trajectory using JointTrajectoryCurve and execute
    jtc = JointTrajectoryCurve()
    traj = jtc.generate_trajectory(path, move_time=5)
    sawyer_robot.execute_trajectory(traj)
    try:
        while True:
            sim.step()
    except KeyboardInterrupt:
        p.disconnect()
        sys.exit(0)


if __name__ == "__main__":
    main()
