import os
import json
import time
from functools import partial
import datetime
import random

import pybullet as p
if os.environ.get('ROS_DISTRO'):
    import rospy
import numpy as np

from cairo_simulator.core.sim_context import SawyerBiasedTSRSimContext
from cairo_simulator.core.utils import ASSETS_PATH

from cairo_planning.collisions import DisabledCollisionsContext
from cairo_planning.local.interpolation import parametric_lerp
from cairo_planning.local.curve import JointTrajectoryCurve
from cairo_planning.planners import LazyCPRM
from cairo_planning.sampling.samplers import HyperballSampler
from cairo_planning.geometric.state_space import SawyerConfigurationSpace
from cairo_planning.core.serialization import load_model


def main():
    # Reload the samples and configuration
    directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data/biased_serialized_model")
    _, samples, graph = load_model(directory)

    start_goal_configs_fpath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data/start_goal_configs.json")
    with open(start_goal_configs_fpath, 'r') as f:
        start_goal_configs = json.load(f)

    config = {}
    config["sim"] = {
            "use_real_time": False
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
    
    initial_start = [0.4512693515323636, 0.578072751043309, -1.7085853204387587, 0.596159706024823, 1.9871449177039127, 1.2134687707559257, -1.569380122838989]

    initial_goal = [-0.5684057726305594, 0.5583954509945905, -1.5247855621059458, 0.7754717976826726, 1.901730705121558, 1.135705090297649, -2.8032179515916686]


    sim_context = SawyerBiasedTSRSimContext(configuration=config)
    sim = sim_context.get_sim_instance()
    logger = sim_context.get_logger()
    sawyer_robot = sim_context.get_robot()
    biased_state_space = sim_context.get_state_space()
    svc = sim_context.get_state_validity()
    tsr = sim_context.get_tsr()
    sawyer_robot.set_joint_state(initial_start)
    time.sleep(3)
    s_g_configs = list(zip(start_goal_configs["start_configurations"], start_goal_configs["start_configurations"]))
    print(s_g_configs)
    ###########
    # CPRM #
    ###########
    # The specific space we sample from is the Hyberball centered at the midpoint between two candidate points. 
    # This is used to bias tree grwoth between two points when using CBiRRT2 as our local planner for a constrained PRM.
    tree_state_space = SawyerConfigurationSpace(sampler=HyperballSampler())
    # Use parametric linear interpolation with 10 steps between points.
    interp = partial(parametric_lerp, steps=10)
    # See params for PRM specific parameters
    prm = LazyCPRM(SawyerBiasedTSRSimContext, config, sawyer_robot, tsr, biased_state_space, tree_state_space, svc, interp, params={
        'n_samples': 3000, 'k': 15, 'planning_attempts': 5, 'ball_radius': 10.0, 'smooth_path': True, 'smoothing_time':10}, tree_params={'iters': 500, 'q_step': .38, 'epsilon': config['tsr']['epsilon'], 'e_step': config['tsr']['e_step']}, logger=logger)
    logger.info("Preloading samples and model....")
    prm.preload(samples, graph)

    random.shuffle(s_g_configs)
    for start, goal in s_g_configs:
        sawyer_robot.set_joint_state(start)
        try:
            with DisabledCollisionsContext(sim, [], []):
                logger.info("Planning.")
                path = prm.plan(np.array(initial_start), np.array(initial_goal))
                # splining uses numpy so needs to be converted
                path = np.array([np.array(p) for p in path])
            
            # Create a MinJerk spline trajectory using JointTrajectoryCurve and execute
            jtc = JointTrajectoryCurve()
            traj = jtc.generate_trajectory(path, move_time=10)
            time.sleep(1)
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
            prm.remove_start_and_end()
        except Exception as e:
            print(e)
       

if __name__ == "__main__":
    main()
