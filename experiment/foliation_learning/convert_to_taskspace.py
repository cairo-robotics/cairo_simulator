import os
import sys
from collections import OrderedDict
from functools import partial
import time
import json

import networkx as nx
from sklearn.mixture import GaussianMixture
import pybullet as p
if os.environ.get('ROS_DISTRO'):
    import rospy
import numpy as np

from cairo_simulator.core.sim_context import SawyerBiasedTSRSimContext
from cairo_simulator.core.utils import ASSETS_PATH

from cairo_planning.collisions import DisabledCollisionsContext
from cairo_planning.local.interpolation import parametric_lerp
from cairo_planning.local.curve import JointTrajectoryCurve
from cairo_planning.planners import CBiRRT2
from cairo_planning.geometric.state_space import DistributionSpace
from cairo_planning.geometric.transformation import quat2rpy
from cairo_planning.sampling.samplers import DistributionSampler
from cairo_planning.constraints.foliation import VGMMFoliationClustering


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
    
    sim_context = SawyerBiasedTSRSimContext(configuration=config)
    sim = sim_context.get_sim_instance()
    logger = sim_context.get_logger()
    # _ = sim_context.get_state_space()
    sawyer_robot = sim_context.get_robot()
    # _ = sawyer_robot.get_simulator_id()
    tsr = sim_context.get_tsr()
    _ = sim_context.get_sim_objects(['Ground'])[0]
    svc = sim_context.get_state_validity()

    # Collect all joint configurations from all demonstration .json files.
    test_configurations = []
    samples_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "samples.json")
  
    with open(samples_file, "r") as f:
        raw_data = json.load(f)
        samples = []
        for sample in raw_data['samples']:
            pos, ori_q = sawyer_robot.solve_forward_kinematics(sample)[0]
            ori_ts = quat2rpy(ori_q)
            samples.append([list(pos), list(ori_ts)])
    
    data = {}
    data["samples"] = samples
    ts_samples_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "taskspace_samples.json")
    with open(ts_samples_file, 'w') as f:
        json.dump(data, f)

if __name__ == "__main__":
    main()