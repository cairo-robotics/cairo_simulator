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
from cairo_planning.sampling.samplers import UniformSampler
from cairo_planning.geometric.state_space import SawyerConfigurationSpace

from cairo_planning.core.serialization import load_model


def main():

    # Reload the samples and configuration
    directory = os.path.join(os.path.dirname(
        os.path.abspath(__file__)), "serialization_data/2021-08-17T16-25-40")
    config, samples, graph = load_model(directory)
    config["sim_objects"] = [
        {
            "object_name": "Ground",
            "model_file_or_sim_id": "plane.urdf",
            "position": [0, 0, 0]
        },
        {
            "object_name": "sphere1",
            "model_file_or_sim_id": 'sphere2.urdf',
            "position": [1.0, -.3, .6],
            "orientation":  [0, 0, 1.5708],
            "fixed_base": 1
        },
        {
            "object_name": "sphere2",
            "model_file_or_sim_id": 'sphere2.urdf',
            "position": [1.0, -.3, 1.65],
            "orientation":  [0, 0, 1.5708],
            "fixed_base": 1
        }
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

    sawyer_robot.move_to_joint_pos(start)
    time.sleep(5)

    with DisabledCollisionsContext(sim, [], []):
        ###########
        # LazyPRM #
        ###########
        # The specific space we sample from is the Hyberball centered at the midpoint between two candidate points.
        # This is used to bias tree grwoth between two points when using CBiRRT2 as our local planner for a constrained PRM.
        tree_state_space = SawyerConfigurationSpace(sampler=UniformSampler())
        # Use parametric linear interpolation with 10 steps between points.
        interp = partial(parametric_lerp, steps=10)
        # See params for PRM specific parameters
        prm = LazyCPRM(SawyerCPRMSimContext, config, sawyer_robot, tsr, planning_space, tree_state_space, svc, interp, params={
            'n_samples': 3000, 'k': 15, 'planning_attempts': 5, 'ball_radius': 10, 'smooth_path': True, 'cbirrt2_sampling_space': 'uniform', 'smoothing_time': 5}, tree_params={'iters': 50, 'q_step': .48, 'e_step': .25})
        logger.info("Planning....")
        prm.preload(samples, graph)
        ptime1 = time.process_time()
        path = prm.plan(np.array(start), np.array(goal))

    # splining uses numpy so needs to be converted
    path = [np.array(p) for p in path]
    # Create a MinJerk spline trajectory using JointTrajectoryCurve and execute
    jtc = JointTrajectoryCurve()
    traj = jtc.generate_trajectory(path, move_time=20)
    ptime2 = time.process_time()
    logger.warn(ptime2 - ptime1)
    sawyer_robot.execute_trajectory(traj)
    try:
        while True:
            sim.step()
    except KeyboardInterrupt:
        p.disconnect()
        sys.exit(0)


if __name__ == "__main__":
    main()
