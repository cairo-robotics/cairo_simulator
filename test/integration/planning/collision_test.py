from collections import OrderedDict
import json
import pprint
import os
import sys
from functools import partial
import time

if os.environ.get('ROS_DISTRO'):
    import rospy
import numpy as np
import networkx as nx

from cairo_simulator.core.sim_context import SawyerSimContext
from cairo_simulator.core.simulator import SimObject
from cairo_simulator.core.primitives import create_box
from cairo_planning.core.planning_context import SawyerPlanningContext
from cairo_planning.collisions import DisabledCollisionsContext
from cairo_planning.local.interpolation import parametric_lerp
from cairo_planning.local.curve import JointTrajectoryCurve
from cairo_planning.planners import LazyPRM


if __name__ == "__main__":
    #############################################
    #       Important Limits for Samplers       #
    #############################################
    limits = [['right_j0', (-3.0503, 3.0503)],
            ['right_j1', (-3.8095, 2.2736)],
            ['right_j2', (-3.0426, 3.0426)],
            ['right_j3', (-3.0439, 3.0439)],
            ['right_j4', (-2.9761, 2.9761)],
            ['right_j5', (-2.9761, 2.9761)],
            ['right_j6', (-4.7124, 4.7124)],
            ['right_gripper_l_finger_joint', (0.0, 0.020833)],
            ['right_gripper_r_finger_joint',
            (-0.020833, 0.0)],
            ['head_pan', (-5.0952, 0.9064)]]


    final_path = []
    sim_context = SawyerSimContext(None, setup=False)
    sim_context.setup(sim_overrides={"run_parallel": False})
    sim = sim_context.get_sim_instance()
    logger = sim_context.get_logger()
    sawyer_robot = sim_context.get_robot()
    interp_fn = partial(parametric_lerp, steps=10)
    # sim_obj = SimObject('test', 'r2d2.urdf', (.6, -0.5, .6), fixed_base=1)
    box = SimObject('box', create_box(w=0.5, l=0.5, h=.6), (.5, -0.5, .6), fixed_base=1)
    state_space = sim_context.get_state_space()
    # note that this call for state validty comes after all the obejcts have been made.
    svc = sim_context.get_state_validity()

    start = [0, 0, 0, 0, 0, 0, -np.pi/2]

    goal = [-1.9622245072067646, 0.8439858364277937, 1.3628459180018329, -
            0.2383928041974519, -2.7327884695211555, -2.2177502341009134, -0.08992133311928363]

    time.sleep(5)

    ####################################
    # SIMULATION AND PLANNING CONTEXTS #
    ####################################
    with DisabledCollisionsContext(sim, [], []):
        #######
        # PRM #
        #######
        # Use parametric linear interpolation with 10 steps between points.
        interp = partial(parametric_lerp, steps=10)
        # See params for PRM specific parameters
        prm = LazyPRM(state_space, svc, interp_fn, params={
                'n_samples': 2000, 'k': 8, 'ball_radius': 2.0})
        logger.info("Planning....")
        plan = prm.plan(np.array(start), np.array(goal))
        # get_path() reuses the interp function to get the path between vertices of a successful plan
        path = prm.get_path(plan)
    if len(path) == 0:
        logger.info("Planning failed....")
        sys.exit(1)
    logger.info("Plan found....")

    # splinging uses numpy so needs to be converted
    path = [np.array(p) for p in path]
    logger.info("Length of path: {}".format(len(path)))
    final_path = final_path + path

    sawyer_robot.move_to_joint_pos(start)
    time.sleep(3)
    while sawyer_robot.check_if_at_position(start, 0.5) is False:
        time.sleep(0.1)
        sim.step()

    key = input("Press any key to excute plan.")

    if len(path) > 0:
        # Create a MinJerk spline trajectory using JointTrajectoryCurve and execute
        jtc = JointTrajectoryCurve()
        traj = jtc.generate_trajectory(final_path, move_time=10)
        sawyer_robot.execute_trajectory(traj)
        try:
            while True:
                sim.step()
        except KeyboardInterrupt:
            sys.exit(0)
    else:
        logger.err("No path found.")

