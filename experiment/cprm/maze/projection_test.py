import os
import sys
import time
import timeit
import json

import pybullet as p
if os.environ.get('ROS_DISTRO'):
    import rospy
import numpy as np

from cairo_simulator.core.sim_context import SawyerTSRSimContext
from cairo_simulator.core.utils import ASSETS_PATH

from cairo_planning.collisions import DisabledCollisionsContext
from cairo_planning.geometric.transformation import quat2rpy
from cairo_planning.constraints.projection import project_config
from cairo_planning.geometric.utils import wrap_to_interval
from cairo_planning.geometric.state_space import DistributionSpace
from cairo_planning.sampling.samplers import DistributionSampler
from cairo_planning.geometric.distribution import KernelDensityDistribution


def main():

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
            "primitive_configs": {"w": .015, "l": .35, "h": .06},
            "sim_object_configs": 
                {
                    "object_name": "center_wall",
                    "position": [.67, 0, .64],
                    "orientation":  [0, 0, 0],
                    "fixed_base": 1    
                }
        },
        {
            "type": "box",
            "primitive_configs": {"w": .015, "l": .2, "h": .06},
            "sim_object_configs": 
                {
                    "object_name": "right_outer_bend",
                    "position": [.79, .25, .64],
                    "orientation":  [0, 0, -np.pi/6],
                    "fixed_base": 1    
                }
        },
        {
            "type": "box",
            "primitive_configs": {"w": .015, "l": .2, "h": .06},
            "sim_object_configs": 
                {
                    "object_name": "right_outer_bend2",
                    "position": [.62, .25, .64],
                    "orientation":  [0, 0, np.pi/6],
                    "fixed_base": 1    
                }
        },
         {
            "type": "box",
            "primitive_configs": {"w": .015, "l": .10, "h": .06},
            "sim_object_configs": 
                {
                    "object_name": "right_inner_bend",
                    "position": [.73, .29, .64],
                    "orientation":  [0, 0, -np.pi/6],
                    "fixed_base": 1    
                }
        },
        {
            "type": "box",
            "primitive_configs": {"w": .015, "l": .10, "h": .06},
            "sim_object_configs": 
                {
                    "object_name": "right_inner_bend2",
                    "position": [.69, .29, .64],
                    "orientation":  [0, 0, np.pi/6],
                    "fixed_base": 1    
                }
        },
        {
            "type": "box",
            "primitive_configs": {"w": .015, "l": .2, "h": .06},
            "sim_object_configs": 
                {
                    "object_name": "left_outer_bend",
                    "position": [.79, -.25, .64],
                    "orientation":  [0, 0, np.pi/6],
                    "fixed_base": 1    
                }
        },
        {
            "type": "box",
            "primitive_configs": {"w": .015, "l": .2, "h": .06},
            "sim_object_configs": 
                {
                    "object_name": "left_outer_bend2",
                    "position": [.62, -.25, .64],
                    "orientation":  [0, 0, -np.pi/6],
                    "fixed_base": 1    
                }
        },
         {
            "type": "box",
            "primitive_configs": {"w": .015, "l": .10, "h": .06},
            "sim_object_configs": 
                {
                    "object_name": "left_inner_bend",
                    "position": [.73, -.29, .64],
                    "orientation":  [0, 0, np.pi/6],
                    "fixed_base": 1    
                }
        },
        {
            "type": "box",
            "primitive_configs": {"w": .015, "l": .10, "h": .06},
            "sim_object_configs": 
                {
                    "object_name": "left_inner_bend2",
                    "position": [.69, -.29, .64],
                    "orientation":  [0, 0, -np.pi/6],
                    "fixed_base": 1    
                }
        },
        {
            "type": "box",
            "primitive_configs": {"w": .015, "l": .35, "h": .06},
            "sim_object_configs": 
                {
                    "object_name": "center_wall2",
                    "position": [.74, 0, .64],
                    "orientation":  [0, 0, 0],
                    "fixed_base": 1    
                }
        },
        ]
            
    config['tsr'] = {
            'degrees': False,
            'epsilon': .07,
            'e_step': .25,
            "T0_w": [.726, 0, .84, 0, 0, 0],
            "Tw_e": [0, 0, 0, np.pi, 0, np.pi/2],
            "Bw": [[(-.12, .12), (-100, 100), (0, .01)],  
                    [(-.001, .001), (-.001, .001), (-.001, .001)]]
    }
    
    sim_context = SawyerTSRSimContext(config)
    sim = sim_context.get_sim_instance()
    _ = sim_context.get_logger()
    scs = sim_context.get_state_space()
    svc = sim_context.get_state_validity()
    sawyer_robot = sim_context.get_robot()
    tsr = sim_context.get_tsr()
    _ = sawyer_robot.get_simulator_id()
    _ = sim_context.get_sim_objects(['Ground'])[0]


            #     config['tsr'] = {
        #     'degrees': False,
        #     "T0_w": [0, 0, .9, 0, 0, 0],
        #     "Tw_e": [0, 0, 0, -3.12266697, 0.02430386, -1.50671032],
        #     "Bw": [[(0, 100), (-100, 100), (-5, 5)],  
        #             [(-.07, .07), (-.07, .07), (-.07, .07)]]
        # }
        
    n_samples = 5
    valid_samples = []
    starttime = timeit.default_timer()

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
    model = KernelDensityDistribution(bandwidth=config['sampling_bias']['bandwidth'])
    model.fit(config['sampling_bias']['data'])
    sampler = DistributionSampler(distribution_model=model, fraction_uniform=config['sampling_bias']['fraction_uniform'])
    state_space = DistributionSpace(sampler=sampler)

    # Disabled collisions during planning with certain eclusions in place.
    with DisabledCollisionsContext(sim, [], []):
        print("Sampling start time is :", starttime)
        while len(valid_samples) < n_samples:
            sample = state_space.sample()
            if svc.validate(sample):
                q_constrained = project_config(sawyer_robot, tsr, np.array(
                sample), np.array(sample), epsilon=.05, e_step=.25)
                normalized_q_constrained = []
                if q_constrained is not None:
                    for value in q_constrained:
                        normalized_q_constrained.append(
                            wrap_to_interval(value))
                else:
                    continue
                if svc.validate(normalized_q_constrained):
                    print(normalized_q_constrained)
                    print()
                    valid_samples.append(normalized_q_constrained)
        print("The time difference is :", timeit.default_timer() - starttime)
        print("{} valid of {}".format(len(valid_samples), n_samples))
    for sample in valid_samples:
        world_pose, local_pose = sawyer_robot.solve_forward_kinematics(sample)
        trans, quat = world_pose[0], world_pose[1]
        print(trans, quat2rpy(quat))
        sawyer_robot.set_joint_state(sample)
        time.sleep(1.5)

    # # Loop until someone shuts us down
    try:
        while True:
            pass
            # sim.step()
    except KeyboardInterrupt:
        p.disconnect()
        sys.exit(0)


if __name__ == "__main__":
    main()
