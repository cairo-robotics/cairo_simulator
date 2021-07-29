import os
from functools import partial
import time
import sys 
import json

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
    directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), "serialization_data/2021-07-29T15-13-50")
    config, samples, graph = load_model(directory)


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
            'n_samples': 3000, 'k': 8, 'planning_attempts': 5, 'ball_radius': 2.0}, tree_params={'iters': 50, 'q_step': .5})
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
    # splining uses numpy so needs to be converted
    path = [np.array(p) for p in path]
    # Create a MinJerk spline trajectory using JointTrajectoryCurve and execute
    jtc = JointTrajectoryCurve()
    traj = jtc.generate_trajectory(path, move_time=5)
    traj_data = {}
    traj_data["trajectory"] = []
    for point in traj:
        traj_point = {
            "time": point[0],
            "point": point[1]
        }
        traj_data['trajectory'].append(traj_point)
    with open('constrained_traj.json', 'w') as f:
        json.dump(traj_data, f)


if __name__ == "__main__":
    main()

