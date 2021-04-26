import os
import sys
from functools import partial
import time

import pybullet as p
if os.environ.get('ROS_DISTRO'):
    import rospy
import numpy as np
import igraph as ig

from cairo_simulator.core.sim_context import SawyerCPRMSimContext
from cairo_simulator.core.simulator import SimObject
from cairo_simulator.core.primitives import create_box
from cairo_simulator.core.utils import ASSETS_PATH

from cairo_planning.collisions import DisabledCollisionsContext
from cairo_planning.local.interpolation import parametric_lerp
from cairo_planning.local.curve import JointTrajectoryCurve
from cairo_planning.planners import CPRM
from cairo_planning.sampling.samplers import HyperballSampler
from cairo_planning.geometric.state_space import SawyerTSRConstrainedSpace, SawyerConfigurationSpace

def main():

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

    config = {}
    config["sim_objects"] = [
        {
            "object_name": "Ground",
            "model_file_or_sim_id": "plane.urdf",
            "position": [0, 0, 0]
        },
        # {
        #     "object_name": "Table",
        #     "model_file_or_sim_id": ASSETS_PATH + 'table.sdf',
        #     "position": [.6, -.8, 1.0],
        #     "orientation":  [0, 0, 1.5708],
        #     "fixed_base": 1
        # }
    ]

    config["tsr"] = {
        'degrees': False,
        "T0_w": [.7, 0, 0, 0, 0, 0],
        "Tw_e": [-.2, 0, 1.0, np.pi/2, 3*np.pi/2, np.pi/2], # level end-effector pointing away from sawyer's "front"
        "Bw": [[[0, 100], [-100, 100], [-100, .3]],  # Cannot go above 1.3 m
              [[-.07, .07], [-.07, .07], [-.07, .07]]]
    }

    sim_context = SawyerCPRMSimContext(config)
    sim = sim_context.get_sim_instance()
    logger = sim_context.get_logger()
    planning_space = sim_context.get_state_space()
    sawyer_robot = sim_context.get_robot()
    tsr = sim_context.get_tsr()
    box = SimObject('box', create_box(w=.5, l=.5, h=.5), (.7, -0.25, .45), fixed_base=1)
    svc = sim_context.get_state_validity()

    start = [0, 0, 0, 0, 0, 0, -np.pi/2]

    goal = [-1.9622245072067646, 0.8439858364277937, 1.3628459180018329, -
            0.2383928041974519, -2.7327884695211555, -2.2177502341009134, -0.08992133311928363]

    # sawyer_robot.move_to_joint_pos(goal)
    # time.sleep(5)
    sawyer_robot.move_to_joint_pos(start)
    time.sleep(5)
    # Utilizes RPY convention

    with DisabledCollisionsContext(sim, [], []):
        #########################
        # Contrained PRM (CPRM) #
        #########################
        # The specific space we sample from is the Hyberball centered at the midpoint between two candidate points. 
        # This is used to bias tree grwoth between two points when using CBiRRT2 as our local planner for a constrained PRM.
        tree_state_space = SawyerConfigurationSpace(sampler=HyperballSampler())
        # Use parametric linear interpolation with 10 steps between points.
        interp = partial(parametric_lerp, steps=10)
        # See params for PRM specific parameters robot, tsr, state_space, state_validity_checker, interpolation_fn, params
        prm = CPRM(SawyerCPRMSimContext, config, sawyer_robot, tsr, planning_space, tree_state_space, svc, interp, params={
            'n_samples': 1200, 'k': 6, 'planning_attempts': 5, 'ball_radius': 2.0}, tree_params={'iters': 50, 'q_step': .5})
        logger.info("Planning....")
        plan = prm.plan(np.array(start), np.array(goal))
        # get_path() reuses the interp function to get the path between vertices of a successful plan
        if plan is not None:
            path = prm.get_path(plan)
        else:
            path = []
    if len(path) == 0:
        visual_style = {}
        visual_style["vertex_color"] = ["blue" if v['name'] in [
            'start', 'goal'] else "white" for v in prm.graph.vs]
        visual_style["bbox"] = (1200, 1200)
        ig.plot(prm.graph, **visual_style)
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
