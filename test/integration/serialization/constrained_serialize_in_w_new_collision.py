import os
from functools import partial
import time
import sys 

import pybullet as p
if os.environ.get('ROS_DISTRO'):
    import rospy
import numpy as np
import igraph as ig

from cairo_simulator.core.sim_context import SawyerCPRMSimContext

from cairo_planning.collisions import DisabledCollisionsContext
from cairo_planning.local.interpolation import parametric_lerp
from cairo_planning.local.curve import JointTrajectoryCurve
from cairo_planning.planners import LazyCPRM
from cairo_planning.sampling.samplers import HyperballSampler
from cairo_planning.geometric.state_space import SawyerConfigurationSpace

from cairo_planning.core.serialization import load_model

def main():

    # Reload the samples and configuration
    directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), "serialization_data/2021-08-03T13-28-14")
    config, samples, graph = load_model(directory)


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
    config["sim_objects"] = [
        {
            "object_name": "Ground",
            "model_file_or_sim_id": "plane.urdf",
            "position": [0, 0, 0]
        },
        {
            "object_name": "sphere",
            "model_file_or_sim_id": 'sphere2.urdf',
            "position": [1.0, -.3, .6],
            "orientation":  [0, 0, 1.5708],
            "fixed_base": 1    
        }
        # {
        #     "object_name": "Table",
        #     "model_file_or_sim_id": ASSETS_PATH + 'table.sdf',
        #     "position": [.8, -.6, .6],
        #     "orientation":  [0, 0, 1.5708],
        #     "fixed_base": 1
        # }
    ]
    sim_context = SawyerCPRMSimContext(configuration=config)
    sim = sim_context.get_sim_instance()
    logger = sim_context.get_logger()
    planning_space = sim_context.get_state_space()
    sawyer_robot = sim_context.get_robot()
    tsr = sim_context.get_tsr()
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
        ###########
        # LazyPRM #
        ###########
        # The specific space we sample from is the Hyberball centered at the midpoint between two candidate points. 
        # This is used to bias tree grwoth between two points when using CBiRRT2 as our local planner for a constrained PRM.
        tree_state_space = SawyerConfigurationSpace(sampler=HyperballSampler())
        # Use parametric linear interpolation with 10 steps between points.
        interp = partial(parametric_lerp, steps=10)
        # See params for PRM specific parameters
        prm = LazyCPRM(SawyerCPRMSimContext, config, sawyer_robot, tsr, planning_space, tree_state_space, svc, interp, params={
            'n_samples': 3000, 'k': 8, 'planning_attempts': 5, 'ball_radius': 2.5}, tree_params={'iters': 50, 'q_step': .5})
        logger.info("Planning....")
        prm.preload(samples, graph)
        path = prm.plan(np.array(start), np.array(goal))


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
    traj = jtc.generate_trajectory(path, move_time=20)
    sawyer_robot.execute_trajectory(traj)
    try:
        while True:
            sim.step()
    except KeyboardInterrupt:
        p.disconnect()
        sys.exit(0)


if __name__ == "__main__":
    main()

