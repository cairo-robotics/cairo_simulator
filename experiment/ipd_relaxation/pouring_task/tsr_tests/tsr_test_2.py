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
from cairo_simulator.core.utils import ASSETS_PATH

from cairo_planning.collisions import DisabledCollisionsContext
from cairo_planning.geometric.transformation import xyzrpy2trans, bounds_matrix,  pose2trans, quat2rpy
from cairo_planning.geometric.tsr import TSR
from cairo_planning.constraints.projection import distance_from_TSR


def distance_to_TSR_config(manipulator, q_s, tsr):
    world_pose, _ = manipulator.solve_forward_kinematics(q_s)
    trans, quat = world_pose[0], world_pose[1]
    T0_s = pose2trans(np.hstack([trans + quat]))
    # generates the task space distance and error/displacement vector
    min_distance_new, x_err = distance_from_TSR(T0_s, tsr)
    return min_distance_new, x_err

def main():

    base_config = {}
    base_config["sim"] = {
            "use_real_time": False
        }

    base_config["logging"] = {
            "handlers": ['logging'],
            "level": "debug"
        }

    base_config["sawyer"] = {
            "robot_name": "sawyer0",
            'urdf_file': ASSETS_PATH + 'sawyer_description/urdf/sawyer_static_mug_combined.urdf',
            "position": [0, 0, 0.9],
            "fixed_base": True
        }

    base_config["sim_objects"] = [
        {
            "object_name": "Ground",
            "model_file_or_sim_id": "plane.urdf",
            "position": [0, 0, 0]
        },
        {
            "object_name": "Table",
            "model_file_or_sim_id": ASSETS_PATH + 'table.sdf',
            "position": [0.75, 0, .1],
            "orientation":  [0, 0, 1.5708],
            "fixed_base": 1
        },
    ]
    base_config["primitives"] = [
        {
            "type": "cylinder",
            "primitive_configs": {"radius": .1, "height": .05},
            "sim_object_configs": 
                {
                    "object_name": "cylinder",
                    "position": [.9, -.57, .6],
                    "orientation":  [0, 0, 0],
                    "fixed_base": 1    
                }
        }
    ]
    
    # For the mug-based URDF of sawyer, we need to exclude links that are in constant self collision for the SVC
    base_config["state_validity"] = {
        "self_collision_exclusions": [("mug", "right_gripper_l_finger"), ("mug", "right_gripper_r_finger")]
    }
    sim_context = SawyerSimContext(base_config)
    sim = sim_context.get_sim_instance()
    _ = sim_context.get_logger()
    scs = sim_context.get_state_space()
    svc = sim_context.get_state_validity()
    sawyer_robot = sim_context.get_robot()
    _ = sawyer_robot.get_simulator_id()
    _ = sim_context.get_sim_objects(['Ground'])[0]


    tsr_config = {
        'degrees': False,
        "T0_w":  [.7968, -.5772, 1.05, np.pi/2, 0,  np.pi/2],
        "Tw_e": [0, 0, 0, 0, 0, 0],
        "Bw": [[(-.1, .1), (-.1, .1), (-100, 100)],  
                [(-100, 100), (-100, 100), (-100, 100)]]
    }
        

    T0_w = xyzrpy2trans(tsr_config['T0_w'], degrees=tsr_config['degrees'])
    Tw_e = xyzrpy2trans(tsr_config['Tw_e'], degrees=tsr_config['degrees'])
    Bw = bounds_matrix(tsr_config['Bw'][0], tsr_config['Bw'][1])
    tsr = TSR(T0_w=T0_w, Tw_e=Tw_e, Bw=Bw, bodyandlink=0, manipindex=16)

    
    sample = [-1.2935242817684087, 0.6871494889588692, -1.1435452680721492, -0.7077175889750391, -0.571226569472242, -0.6229696369014182, 2.8631755973215913]
    world_pose, _ = sawyer_robot.solve_forward_kinematics(sample)
    trans, quat = world_pose[0], world_pose[1]
    print(trans, quat2rpy(quat))
    print(tsr_config["T0_w"][0] - trans[0], tsr_config["T0_w"][1] - trans[1], tsr_config["T0_w"][2] - trans[2])
    sawyer_robot.set_joint_state(sample)
    # Disabled collisions during planning with certain eclusions in place.
    with DisabledCollisionsContext(sim, [], []):
        err, _ = distance_to_TSR_config(sawyer_robot, sample, tsr)
        print(err)
        print(_)
    time.sleep(5)

  

if __name__ == "__main__":
    main()
