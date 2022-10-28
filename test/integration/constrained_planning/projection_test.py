import os
import sys
import time
import timeit
import json

import pybullet as p
if os.environ.get('ROS_DISTRO'):
    import rospy
import numpy as np

from cairo_simulator.core.sim_context import SawyerSimContext
from cairo_planning.core.planning_context import SawyerPlanningContext
from cairo_simulator.core.utils import ASSETS_PATH
from cairo_simulator.core.link import get_joint_info_by_name

from cairo_planning.collisions import DisabledCollisionsContext
from cairo_planning.geometric.transformation import xyzrpy2trans, bounds_matrix, quat2rpy
from cairo_planning.geometric.state_space import SawyerTSRConstrainedSpace
from cairo_planning.sampling.samplers import UniformSampler
from cairo_planning.constraints.projection import project_config
from cairo_planning.geometric.tsr import TSR
from cairo_planning.geometric.utils import geodesic_distance, wrap_to_interval


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
    sim_context = SawyerSimContext(config)
    sim = sim_context.get_sim_instance()
    _ = sim_context.get_logger()
    scs = sim_context.get_state_space()
    svc = sim_context.get_state_validity()
    sawyer_robot = sim_context.get_robot()
    _ = sawyer_robot.get_simulator_id()
    _ = sim_context.get_sim_objects(['Ground'])[0]


            #     config['tsr'] = {
        #     'degrees': False,
        #     "T0_w": [0, 0, .9, 0, 0, 0],
        #     "Tw_e": [0, 0, 0, -3.12266697, 0.02430386, -1.50671032],
        #     "Bw": [[(0, 100), (-100, 100), (-5, 5)],  
        #             [(-.07, .07), (-.07, .07), (-.07, .07)]]
        # }
        

         # Collect all joint configurations from all demonstration .json files.
    configurations = []
    # data_directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data/sampling_bias")
  
    # print("Running biased sampling test for {}".format(data_directory))
    # for json_file in os.listdir(data_directory):
    #     filename = os.path.join(data_directory, json_file)
    #     with open(filename, "r") as f:
    #         data = json.load(f)
    #         for entry in data:
    #             configurations.append(entry['robot']['joint_angle'])

    n_samples = 5
    valid_samples = []
    starttime = timeit.default_timer()
    TSR_2_config = {
        'degrees': False,
        "T0_w":  [0.62, -0.6324, 0.15, np.pi/2, -np.pi/2, np.pi/2],
        "Tw_e": [0, 0, 0, 0, 0, 0],
        "Bw": [[(-.02, .02), (-.02, .02), (-100, 100)],  
                [(-100, 100), (-100, 100), (-100, 100)]]
    }
    # Utilizes RPY convention
    T0_w = xyzrpy2trans([0.62, -0.6324, 0.15, np.pi/2, -np.pi/2, np.pi/2], degrees=False)

    # Utilizes RPY convention
    Tw_e = xyzrpy2trans([0, 0, 0, 0, 0, 0], degrees=False)
    
    # Utilizes RPY convention
    Bw = bounds_matrix([(-.02, .02), (-.02, .02), (-100, 100)],  
                [(-100, 100), (-100, 100), (-100, 100)])  # any rotation about z, with limited rotation about x, and y.
    tsr = TSR(T0_w=T0_w, Tw_e=Tw_e, Bw=Bw,
              manipindex=0, bodyandlink=16)
    time.sleep(5)
    start = [-1.0875129593901391, 0.43221681740571016, -1.434156290269138, -0.014856843070199854, -0.13925492871100298, -1.3577540634638976, -0.2363667344199918] 
    test_sample = [-1.3848179487482657, 0.7146040961230544, -1.248842068927987, -0.6474293590198705, -0.5574993249372158, -0.6181778654027106, 2.8736703718739705]
    # Disabled collisions during planning with certain eclusions in place.
    with DisabledCollisionsContext(sim, [], []):
        print("Sampling start time is :", starttime)
        while len(valid_samples) < n_samples:
            # sample = scs.sample()
            sample = test_sample
            if svc.validate(sample):
                q_constrained = project_config(sawyer_robot, tsr, np.array(
                sample), np.array(sample), epsilon=.1, e_step=.25, ignore_termination_condtions=True, iter_count=500)
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
