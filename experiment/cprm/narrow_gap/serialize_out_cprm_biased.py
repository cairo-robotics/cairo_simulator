import os
import json
import time
from functools import partial
import datetime

import pybullet as p

from cairo_planning.geometric import state_space
if os.environ.get('ROS_DISTRO'):
    import rospy
import numpy as np

from cairo_simulator.core.sim_context import SawyerTSRSimContext
from cairo_simulator.core.utils import ASSETS_PATH

from cairo_planning.geometric.state_space import DistributionSpace, SawyerConfigurationSpace


from cairo_planning.collisions import DisabledCollisionsContext
from cairo_planning.local.interpolation import parametric_lerp
from cairo_planning.planners import CPRM
from cairo_planning.sampling.samplers import HyperballSampler, DistributionSampler
from cairo_planning.geometric.distribution import KernelDensityDistribution
from cairo_planning.core.serialization import dump_model

def main():
    
    NUM_SAMPLES = 10

    config = {}
    config["sim"] = {
            "use_real_time": False
        }
    config["sim_objects"] = [
        {
            "object_name": "Ground",
            "model_file_or_sim_id": "plane.urdf",
            "position": [0, 0, 0]
        },
        {
            "object_name": "Table",
            "model_file_or_sim_id": ASSETS_PATH + 'table.sdf',
            "position": [0.9, 0, 0],
            "orientation":  [0, 0, 1.5708]
        },
    ]
    config["primitives"] = [
        {
            "type": "box",
            "primitive_configs": {"w": .2, "l": .45, "h": .35},
            "sim_object_configs": 
                {
                    "object_name": "box",
                    "position": [.6, 0, .7],
                    "orientation":  [0, 0, 0],
                    "fixed_base": 1    
                }
        },
        {
            "type": "box",
            "primitive_configs": {"w": .2, "l": .45, "h": .35},
            "sim_object_configs": 
                {
                    "object_name": "box",
                    "position": [.9, 0, .7],
                    "orientation":  [0, 0, 0],
                    "fixed_base": 1    
                }
        },
        ]
            
    config['tsr'] = {
            'degrees': False,
            "T0_w": [.7, 0, 0, 0, 0, 0],
            "Tw_e": [-.2, 0, .739, -3.1261701132911655, 0.023551837572146628, 0.060331404738664496],
            "Bw": [[(0, 100), (-100, 100), (-.1, 0)],  
                    [(-.07, .07), (-.07, .07), (-.07, .07)]]
        }

    start = [0.4512693515323636, 0.578072751043309, -1.7085853204387587, 0.596159706024823, 1.9871449177039127, 1.2134687707559257, -1.569380122838989]

    goal = [-0.5684057726305594, 0.5583954509945905, -1.5247855621059458, 0.7754717976826726, 1.901730705121558, 1.135705090297649, -2.8032179515916686]


    # Collect all joint configurations from all demonstration .json files.
    configurations = []
    data_directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data/sampling_bias")
  
    print("Running biased sampling test for {}".format(data_directory))
    for json_file in os.listdir(data_directory):
        filename = os.path.join(data_directory, json_file)
        with open(filename, "r") as f:
            data = json.load(f)
            for entry in data:
                configurations.append(entry['robot']['joint_angle'])

    config['sampling_bias'] = {
        'bandwidth': .3,
        'fraction_uniform': .65,
        'data': configurations
    }

    sim_context = SawyerTSRSimContext(configuration=config)
    sim = sim_context.get_sim_instance()
    logger = sim_context.get_logger()
    sawyer_robot = sim_context.get_robot()
    state_space = sim_context.get_state_space()
    svc = sim_context.get_state_validity()
    tsr = sim_context.get_tsr()
    sawyer_robot.set_joint_state(start)
    time.sleep(3)
    # Utilizes RPY convention
    with DisabledCollisionsContext(sim, [], []):
        ###########
        # CPRM #
        ###########
        # The specific space we sample from is the Hyberball centered at the midpoint between two candidate points.
        # This is used to bias tree grwoth between two points when using CBiRRT2 as our local planner for a constrained PRM.
        tree_state_space = SawyerConfigurationSpace(sampler=HyperballSampler())
        # Use parametric linear interpolation with 10 steps between points.
        interp = partial(parametric_lerp, steps=10)
        # See params for PRM specific parameters
        prm = CPRM(SawyerTSRSimContext, config, sawyer_robot, tsr, state_space, tree_state_space, svc, interp, params={
            'n_samples': 25000, 'k': 15, 'planning_attempts': 5, 'ball_radius': 10.0, 'smooth_path': True, 'cbirrt2_sampling_space': 'hyperball', 'smoothing_time': 5}, tree_params={'iters': 50, 'q_step': .48, 'e_step': .25})
        logger.info("Planning....")
        ptime1 = time.process_time()
        path = prm.generate_roadmap(np.array(start), np.array(goal))
       # Dump thje samples and configuration
    dump_model(sim_context.config, prm, os.path.dirname(os.path.abspath(__file__)))
   

if __name__ == "__main__":
    main()


 