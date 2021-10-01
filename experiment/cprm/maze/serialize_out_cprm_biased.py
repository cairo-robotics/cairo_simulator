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

from cairo_simulator.core.sim_context import SawyerBiasedTSRSimContext
from cairo_simulator.core.utils import ASSETS_PATH

from cairo_planning.geometric.state_space import DistributionSpace, SawyerConfigurationSpace


from cairo_planning.collisions import DisabledCollisionsContext
from cairo_planning.local.interpolation import parametric_lerp
from cairo_planning.planners import CPRM
from cairo_planning.sampling.samplers import HyperballSampler, DistributionSampler
from cairo_planning.geometric.distribution import KernelDensityDistribution
from cairo_planning.core.serialization import dump_model

def main():
    
    config = {}
    config["sim"] = {
            "use_real_time": False,
            "use_gui": False
        }

    config["logging"] = {
            "handlers": ['logging'],
            "level": "debug"
        }

    config["sawyer"] = {
            "robot_name": "sawyer0",
            'urdf_file': ASSETS_PATH + 'sawyer_description/urdf/sawyer_static_blockcombine.urdf',
            "position": [0, 0, 0.9],
            "fixed_base": True
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
            "position": [0.6, 0, .1],
            "orientation":  [0, 0, 1.5708]
        },
    ]
    config["primitives"] = [
        {
            "type": "box",
            "primitive_configs": {"w": .015, "l": .35, "h": .08},
            "sim_object_configs": 
                {
                    "object_name": "center_wall",
                    "position": [.62, 0, .64],
                    "orientation":  [0, 0, 0],
                    "fixed_base": 1    
                }
        },
        {
            "type": "box",
            "primitive_configs": {"w": .015, "l": .2, "h": .08},
            "sim_object_configs": 
                {
                    "object_name": "right_outer_bend",
                    "position": [.74, .25, .64],
                    "orientation":  [0, 0, -np.pi/6],
                    "fixed_base": 1    
                }
        },
        {
            "type": "box",
            "primitive_configs": {"w": .015, "l": .2, "h": .08},
            "sim_object_configs": 
                {
                    "object_name": "right_outer_bend2",
                    "position": [.57, .25, .64],
                    "orientation":  [0, 0, np.pi/6],
                    "fixed_base": 1    
                }
        },
         {
            "type": "box",
            "primitive_configs": {"w": .015, "l": .10, "h": .08},
            "sim_object_configs": 
                {
                    "object_name": "right_inner_bend",
                    "position": [.68, .29, .64],
                    "orientation":  [0, 0, -np.pi/6],
                    "fixed_base": 1    
                }
        },
        {
            "type": "box",
            "primitive_configs": {"w": .015, "l": .10, "h": .08},
            "sim_object_configs": 
                {
                    "object_name": "right_inner_bend2",
                    "position": [.64, .29, .64],
                    "orientation":  [0, 0, np.pi/6],
                    "fixed_base": 1    
                }
        },
        {
            "type": "box",
            "primitive_configs": {"w": .015, "l": .2, "h": .08},
            "sim_object_configs": 
                {
                    "object_name": "left_outer_bend",
                    "position": [.74, -.25, .64],
                    "orientation":  [0, 0, np.pi/6],
                    "fixed_base": 1    
                }
        },
        {
            "type": "box",
            "primitive_configs": {"w": .015, "l": .2, "h": .08},
            "sim_object_configs": 
                {
                    "object_name": "left_outer_bend2",
                    "position": [.57, -.25, .64],
                    "orientation":  [0, 0, -np.pi/6],
                    "fixed_base": 1    
                }
        },
         {
            "type": "box",
            "primitive_configs": {"w": .015, "l": .10, "h": .08},
            "sim_object_configs": 
                {
                    "object_name": "left_inner_bend",
                    "position": [.68, -.29, .64],
                    "orientation":  [0, 0, np.pi/6],
                    "fixed_base": 1    
                }
        },
        {
            "type": "box",
            "primitive_configs": {"w": .015, "l": .10, "h": .08},
            "sim_object_configs": 
                {
                    "object_name": "left_inner_bend2",
                    "position": [.64, -.29, .64],
                    "orientation":  [0, 0, -np.pi/6],
                    "fixed_base": 1    
                }
        },
        {
            "type": "box",
            "primitive_configs": {"w": .015, "l": .35, "h": .08},
            "sim_object_configs": 
                {
                    "object_name": "center_wall2",
                    "position": [.69, 0, .64],
                    "orientation":  [0, 0, 0],
                    "fixed_base": 1    
                }
        },
        ]
            
    config['tsr'] = {
            'degrees': False,
            'epsilon': .05,
            'e_step': .25,
            "T0_w": [.676, 0, .84, 0, 0, 0],
            "Tw_e": [0, 0, 0, np.pi, 0, np.pi/2],
            "Bw": [[(-.1, .1), (-100, 100), (0, .005)],  
                    [(-.001, .001), (-.001, .001), (-.001, .001)]]
    }
    
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
        'bandwidth': .1,
        'fraction_uniform': .25,
        'data': configurations
    }

    start = [0.4512693515323636, 0.578072751043309, -1.7085853204387587, 0.596159706024823, 1.9871449177039127, 1.2134687707559257, -1.569380122838989]

    goal = [-0.5684057726305594, 0.5583954509945905, -1.5247855621059458, 0.7754717976826726, 1.901730705121558, 1.135705090297649, -2.8032179515916686]

    sim_context = SawyerBiasedTSRSimContext(configuration=config)
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
        # Use parametric linear interpolation with 25 steps between points.
        interp = partial(parametric_lerp, steps=25)
        # See params for PRM specific parameters
        prm = CPRM(SawyerBiasedTSRSimContext, config, sawyer_robot, tsr, state_space, tree_state_space, svc, interp, params={
            'n_samples': 20000, 'k': 15, 'planning_attempts': 5, 'ball_radius': 10.0, 'smooth_path': True, 'cbirrt2_sampling_space': 'hyperball', 'smoothing_time': 5}, tree_params={'iters': 50, 'q_step': .38, 'epsilon': config['tsr']['epsilon'], 'e_step': config['tsr']['e_step']})
        logger.info("Planning....")
        ptime1 = time.process_time()
        path = prm.generate_roadmap(np.array(start), np.array(goal))
       # Dump thje samples and configuration
    dump_model(sim_context.config, prm, os.path.dirname(os.path.abspath(__file__)))
   

if __name__ == "__main__":
    main()


 