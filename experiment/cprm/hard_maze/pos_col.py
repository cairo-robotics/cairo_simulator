import os
import time


import pybullet as p
if os.environ.get('ROS_DISTRO'):
    import rospy
import numpy as np

from cairo_simulator.core.sim_context import SawyerTSRSimContext
from cairo_simulator.core.utils import ASSETS_PATH
from cairo_planning.collisions import DisabledCollisionsContext
from cairo_planning.geometric.transformation import quat2rpy


global dist, inc
inc = 3
dist = .025

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
            
            

    start = [0.4512693515323636, 0.578072751043309, -1.7085853204387587, 0.596159706024823, 1.9871449177039127, 1.2134687707559257, -1.569380122838989]
    

    sim_context = SawyerTSRSimContext(configuration=config)
    sim = sim_context.get_sim_instance()
    logger = sim_context.get_logger()
    # _ = sim_context.get_state_space()
    sawyer_robot = sim_context.get_robot()
    # _ = sawyer_robot.get_simulator_id()
    _ = sim_context.get_sim_objects(['Ground'])[0]
    svc = sim_context.get_state_validity()

    sawyer_robot.set_joint_state(start)
    time.sleep(5)

    key_u = ord('u') #y up
    key_h = ord('h') #x down
    key_j = ord('j') #y down
    key_k = ord('k') #x up

    key_o = ord('o') #z up
    key_l = ord('l') #z downkey_8 = ord('8')
    key_d = ord('d')
    key_q = ord('q')

    keys = p.getKeyboardEvents()
    joint_config = sawyer_robot.get_current_joint_states()[0:7]
    print(joint_config)
    fk_results = sawyer_robot.solve_forward_kinematics(joint_configuration=joint_config)
    print(fk_results)
    xcurrent, ycurrent, zcurrent = fk_results[0][0]
    orientation = fk_results[0][1]
    input("Press any key to continue")
    increment = [.0025, .025, .25]
    def print_info(joint_config):
        print(joint_config)
        result = sawyer_robot.solve_forward_kinematics(joint_configuration=joint_config)
        print(result)
        print(quat2rpy(result[0][1]))
        print(svc.validate(joint_config))
        print()
    i = 3
    global dist
    def set_distance():
        global dist
        global inc
        dist = increment[inc % 3]
        inc += 1
        print("DISTANCE IS NOW: {}".format(dist))

    with DisabledCollisionsContext(sim, [], [], disable_visualization=False):
        while key_q not in keys:
            keys = p.getKeyboardEvents()
            if key_k in keys and keys[key_k] & p.KEY_WAS_TRIGGERED:
                ycurrent+=dist
                xyz = [xcurrent, ycurrent, zcurrent]
                joint_config = sawyer_robot.solve_inverse_kinematics(xyz, orientation)
                sawyer_robot.move_to_joint_pos(joint_config)
                print_info(joint_config)
            if key_u in keys and keys[key_u] & p.KEY_WAS_TRIGGERED:
                xcurrent-=dist
                xyz = [xcurrent, ycurrent, zcurrent]
                joint_config = sawyer_robot.solve_inverse_kinematics(xyz, orientation)
                sawyer_robot.move_to_joint_pos(joint_config)
                print_info(joint_config)
            if key_h in keys and keys[key_h] & p.KEY_WAS_TRIGGERED:
                ycurrent-=dist
                xyz = [xcurrent, ycurrent, zcurrent]
                joint_config = sawyer_robot.solve_inverse_kinematics(xyz, orientation)
                sawyer_robot.move_to_joint_pos(joint_config)
                print_info(joint_config)
            if key_j in keys and keys[key_j] & p.KEY_WAS_TRIGGERED:
                xcurrent+=dist
                xyz = [xcurrent, ycurrent, zcurrent]
                joint_config = sawyer_robot.solve_inverse_kinematics(xyz, orientation)
                sawyer_robot.move_to_joint_pos(joint_config)
                print_info(joint_config)
            if key_o in keys and keys[key_o] & p.KEY_WAS_TRIGGERED:
                zcurrent+=dist
                xyz = [xcurrent, ycurrent, zcurrent]
                joint_config = sawyer_robot.solve_inverse_kinematics(xyz, orientation)
                sawyer_robot.move_to_joint_pos(joint_config)
                print_info(joint_config)
            if key_l in keys and keys[key_l] & p.KEY_WAS_TRIGGERED:
                zcurrent-=dist
                xyz = [xcurrent, ycurrent, zcurrent]
                joint_config = sawyer_robot.solve_inverse_kinematics(xyz, orientation)
                sawyer_robot.move_to_joint_pos(joint_config)
                print_info(joint_config)
            if key_d in keys and keys[key_d] & p.KEY_WAS_TRIGGERED:
                set_distance()
            if key_q in keys and keys[key_q] & p.KEY_WAS_TRIGGERED:
                break
            

       


if __name__ == "__main__":
    main()
