import os
import time


import pybullet as p
if os.environ.get('ROS_DISTRO'):
    import rospy
import numpy as np

from cairo_simulator.core.sim_context import SawyerCPRMSimContext
from cairo_simulator.core.utils import ASSETS_PATH
from cairo_planning.collisions import DisabledCollisionsContext




def main():

    config = {}
    config = {}
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
            

    start = [
        0.673578125,
        -0.2995908203125,
        -0.21482421875,
        1.4868740234375,
        0.53829296875,
        0.4117080078125,
        -1.2169501953125]
    

    sim_context = SawyerCPRMSimContext(configuration=config)
    sim = sim_context.get_sim_instance()
    logger = sim_context.get_logger()
    # _ = sim_context.get_state_space()
    sawyer_robot = sim_context.get_robot()
    # _ = sawyer_robot.get_simulator_id()
    _ = sim_context.get_sim_objects(['Ground'])[0]
    svc = sim_context.get_state_validity()

    sawyer_robot.move_to_joint_pos(start)
    time.sleep(5)

    key_w = ord('u') #y up
    key_a = ord('h') #x down
    key_s = ord('j') #y down
    key_d = ord('k') #x up

    key_o = ord('o') #z up
    key_l = ord('l') #z downkey_8 = ord('8')

    key_q = ord('q')
    distancechange=0.025
    keys = p.getKeyboardEvents()
    joint_config = sawyer_robot.get_current_joint_states()[0:7]
    print(joint_config)
    fk_results = sawyer_robot.solve_forward_kinematics(joint_configuration=joint_config)
    print(fk_results)
    xcurrent, ycurrent, zcurrent = fk_results[0][0]
    orientation = fk_results[0][1]
    input("Press any key to continue")
    with DisabledCollisionsContext(sim, [], [], disable_visualization=False):
        while key_q not in keys:
            xoffset=0
            yoffset=0
            print(keys)
            keys = p.getKeyboardEvents()
            if key_s in keys and keys[key_s] & p.KEY_WAS_TRIGGERED:
                ycurrent+=distancechange
                xyz = [xcurrent, ycurrent, zcurrent]
                joint_config = sawyer_robot.solve_inverse_kinematics(xyz, orientation)
                sawyer_robot.move_to_joint_pos(joint_config)
            if key_a in keys and keys[key_a] & p.KEY_WAS_TRIGGERED:
                xcurrent-=distancechange
                xyz = [xcurrent, ycurrent, zcurrent]
                joint_config = sawyer_robot.solve_inverse_kinematics(xyz, orientation)
                sawyer_robot.move_to_joint_pos(joint_config)
            if key_w in keys and keys[key_w] & p.KEY_WAS_TRIGGERED:
                ycurrent-=distancechange
                xyz = [xcurrent, ycurrent, zcurrent]
                joint_config = sawyer_robot.solve_inverse_kinematics(xyz, orientation)
                sawyer_robot.move_to_joint_pos(joint_config)
            if key_d in keys and keys[key_d] & p.KEY_WAS_TRIGGERED:
                xcurrent+=distancechange
                xyz = [xcurrent, ycurrent, zcurrent]
                joint_config = sawyer_robot.solve_inverse_kinematics(xyz, orientation)
                sawyer_robot.move_to_joint_pos(joint_config)
            if key_o in keys and keys[key_o] & p.KEY_WAS_TRIGGERED:
                zcurrent+=distancechange
                xyz = [xcurrent, ycurrent, zcurrent]
                joint_config = sawyer_robot.solve_inverse_kinematics(xyz, orientation)
                sawyer_robot.move_to_joint_pos(joint_config)
            if key_l in keys and keys[key_l] & p.KEY_WAS_TRIGGERED:
                zcurrent-=distancechange
                xyz = [xcurrent, ycurrent, zcurrent]
                joint_config = sawyer_robot.solve_inverse_kinematics(xyz, orientation)
                sawyer_robot.move_to_joint_pos(joint_config)  
            if key_q in keys and keys[key_q] & p.KEY_WAS_TRIGGERED:
                break
            print(sawyer_robot.solve_forward_kinematics(joint_configuration=joint_config))
            print(svc.validate(joint_config))

       


if __name__ == "__main__":
    main()
