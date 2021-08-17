import os
from functools import partial
import time

import pybullet as p
if os.environ.get('ROS_DISTRO'):
    import rospy
import numpy as np

from cairo_simulator.core.sim_context import SawyerCPRMSimContext

from cairo_planning.collisions import DisabledCollisionsContext
from cairo_planning.local.interpolation import parametric_lerp
from cairo_planning.local.curve import JointTrajectoryCurve
from cairo_planning.planners import LazyCPRM
from cairo_planning.sampling.samplers import HyperballSampler
from cairo_planning.geometric.state_space import SawyerConfigurationSpace

from cairo_planning.core.serialization import dump_model

def main():

    config = {}
    config["sim_objects"] = [
        {
            "object_name": "Ground",
            "model_file_or_sim_id": "plane.urdf",
            "position": [0, 0, 0]
        },
        # {
        #     "object_name": "sphere",
        #     "model_file_or_sim_id": 'sphere2.urdf',
        #     "position": [1.0, -.3, .6],
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
    svc = sim_context.get_state_validity()

    start = [0, 0, 0, 0, 0, 0, -np.pi/2]

    goal = [-1.9622245072067646, 0.8439858364277937, 1.3628459180018329, -
            0.2383928041974519, -2.7327884695211555, -2.2177502341009134, -0.08992133311928363]

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
            'n_samples': 12000, 'k': 15, 'planning_attempts': 5, 'ball_radius': 10}, tree_params={'iters': 50, 'q_step': .5, 'e_step': .25})
        logger.info("Planning....")
        path = prm.generate_roadmap(np.array(start), np.array(goal))

    # Dump thje samples and configuration
    dump_model(sim_context.config, prm, os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":
    main()
