import os
import time
from pathlib import Path

import pybullet as p

if os.environ.get('ROS_DISTRO'):
    import rospy

import numpy as np
from cairo_planning.collisions import DisabledCollisionsContext
from cairo_planning.constraints.projection import distance_from_TSR
from cairo_planning.geometric.transformation import (bounds_matrix, pose2trans,
                                                     quat2rpy, xyzrpy2trans)
from cairo_planning.geometric.tsr import TSR
from cairo_planning.local.interpolation import parametric_lerp
from cairo_planning_core import Agent
from cairo_simulator.core.sim_context import SawyerBiasedSimContext
from cairo_simulator.core.utils import ASSETS_PATH


def distance_to_TSR_config(manipulator, q_s, tsr):
    world_pose, _ = manipulator.solve_forward_kinematics(q_s)
    trans, quat = world_pose[0], world_pose[1]
    T0_s = pose2trans(np.hstack([trans + quat]))
    # generates the task space distance and error/displacement vector
    min_distance_new, x_err = distance_from_TSR(T0_s, tsr)
    return min_distance_new, x_err

def distance_to_TSR_config_rusty(manipulator, q_s, tsr):
    world_pose = manipulator.forward_kinematics(q_s)
    trans, quat = world_pose[0], world_pose[1]
    T0_s = pose2trans(np.hstack([trans + quat]))
    # generates the task space distance and error/displacement vector
    min_distance_new, x_err = distance_from_TSR(T0_s, tsr)
    return min_distance_new, x_err


global dist, inc
inc = 3
dist = .025

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
            "position": [0, 0, 0],
            "fixed_base": True
        }

    base_config["sim_objects"] = [
        {
            "object_name": "Ground",
            "model_file_or_sim_id": "plane.urdf",
            "position": [0, 0, -.9]
        },
        {
            "object_name": "Table",
            "model_file_or_sim_id": ASSETS_PATH + 'table.sdf',
            "position": [0.75, 0, -.8],
            "orientation":  [0, 0, 1.5708],
            "fixed_base": 1
        },
    ]
    base_config["primitives"] = [
        {
            "type": "cylinder",
            "primitive_configs": {"radius": .12, "height": .05},
            "sim_object_configs": 
                {
                    "object_name": "cylinder",
                    "position": [.85, -.65, -.3],
                    "orientation":  [0, 0, 0],
                    "fixed_base": 1    
                }
        },
        {
            "type": "box",
            "primitive_configs": {"w": .25, "l": .25, "h": .45},
            "sim_object_configs": 
                {
                    "object_name": "box",
                    "position": [.85, -.34, -.2],
                    "orientation":  [0, 0, 0],
                    "fixed_base": 1    
                }
        }
    ]
    
    # For the mug-based URDF of sawyer, we need to exclude links that are in constant self collision for the SVC
    base_config["state_validity"] = {
        "self_collision_exclusions": [("mug", "right_gripper_l_finger"), ("mug", "right_gripper_r_finger")]
    }

    base_config["tsr"] = {
        'degrees': False,
        "T0_w":  [0.62, -0.62, 0.0, np.pi/2, -np.pi/2, np.pi/2],
        "Tw_e": [0, 0, 0, 0, 0, 0],
        "Bw": [[(-.05, .05), (-.05, .05), (-100, 100)],  
                [(-100, 100), (-100, 100), (-100, 100)]]
    }

    
    tsr_config = base_config["tsr"]
    T0_w2 = xyzrpy2trans(tsr_config['T0_w'], degrees=tsr_config['degrees'])
    Tw_e2 = xyzrpy2trans(tsr_config['Tw_e'], degrees=tsr_config['degrees'])
    Bw2 = bounds_matrix(tsr_config['Bw'][0], tsr_config['Bw'][1])
    test_tsr = TSR(T0_w=T0_w2, Tw_e=Tw_e2, Bw=Bw2)
    start = [-1.0618036244451243, 0.45822100278096745, -1.3590412831532155, 0.06518820766705025, -0.04122966871185518, -1.3236995932805757, -0.3422452195554553] 
    end = [-1.2929736023313356, 0.7067918527657278, -1.2365559825555703, -0.6730160377818795, -0.5394501624390595, -0.6562412771070933, 2.8718318164570995]
    sim_context = SawyerBiasedSimContext(configuration=base_config)
    sim = sim_context.get_sim_instance()
    logger = sim_context.get_logger()
    # _ = sim_context.get_state_space()
    sawyer_robot = sim_context.get_robot()
    # _ = sawyer_robot.get_simulator_id()
    _ = sim_context.get_sim_objects(['Ground'])[0]
    svc = sim_context.get_state_validity()
    
    rusty_agent_settings_path = str(
            Path(__file__).parent.absolute()) + "/../settings.yaml"
    print(rusty_agent_settings_path)
    rusty_sawyer_robot = Agent(rusty_agent_settings_path, False, False)
    
    time.sleep(2)
    while True:
        print("Moving to start")
        sawyer_robot.set_joint_state(start)
        print(svc.validate(start))
        fk_results = sawyer_robot.solve_forward_kinematics(joint_configuration=start)
        print(fk_results)
        print(distance_to_TSR_config(sawyer_robot, start, test_tsr))

        rusty_fk_results = rusty_sawyer_robot.forward_kinematics(start)
        print(rusty_fk_results)
        print(distance_to_TSR_config_rusty(rusty_sawyer_robot, start, test_tsr))
        key = input("Press any key to switch to end position, or c to continue")
        if key == 'c':
            break
        print("Moving to end")
        sawyer_robot.set_joint_state(end)
        print(svc.validate(end))
        print(distance_to_TSR_config(sawyer_robot, end, test_tsr))
        fk_results = sawyer_robot.solve_forward_kinematics(joint_configuration=end)
        print(fk_results)
        rusty_fk_results = rusty_sawyer_robot.forward_kinematics(end)
        print(rusty_fk_results)
        print(distance_to_TSR_config_rusty(rusty_sawyer_robot, end, test_tsr))
        key = input("Press any key to switch to start position, or c to continue")
        if key == 'c':
            break
    # time.sleep(5)
    # sawyer_robot.set_joint_state([-1.53672197,  0.58640682, -1.39922472, -0.84949547, -0.52335235,
    #    -0.71242567, -3.25788548])
    # time.sleep(5)
    # sawyer_robot.set_joint_state([-1.12846265,  0.69614649, -1.01425267, -0.73769733, -0.34855334,
    #    -0.65376286, -3.40574205])
    # time.sleep(5)
    # sawyer_robot.set_joint_state([-1.4267265 ,  0.19968268, -1.05435133, -0.66510546, -0.41298996,
    #    -0.5530293 , -3.41016161])
    # time.sleep(5)
    # sawyer_robot.set_joint_state([-1.29645937,  0.58280675, -1.33026263, -0.39615308, -0.71812744,
    #    -0.56107295, -3.60860816])
    with DisabledCollisionsContext(sim, [], [], disable_visualization=False):
        while True:
            sawyer_robot.set_joint_state(start)
            key = input("Press i for an interpolated movement or c to continue")
            if key == 'c':
                break
            if key == 'i':
                steps = parametric_lerp(np.array(start), np.array(end), 100)
                print(steps)
                for i, point in enumerate(steps):
                    if not svc.validate(point):
                        print("Invalid point: {}".format(point))
                        continue
                    sawyer_robot.set_joint_state(point)
                    print(sawyer_robot.solve_forward_kinematics(point)[0][0],  quat2rpy(sawyer_robot.solve_forward_kinematics(point)[0][1]))
                    print(distance_to_TSR_config(sawyer_robot, point, test_tsr))
                    time.sleep(.1)
        sawyer_robot.set_joint_state(end)
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
            print(distance_to_TSR_config(sawyer_robot, start, test_tsr))
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
                    print("TSR VALID: {}".format(test_tsr.is_valid(xyz+list(quat2rpy(orientation)))))
                    print_info(joint_config)
                if key_u in keys and keys[key_u] & p.KEY_WAS_TRIGGERED:
                    xcurrent-=dist
                    xyz = [xcurrent, ycurrent, zcurrent]
                    joint_config = sawyer_robot.solve_inverse_kinematics(xyz, orientation)
                    sawyer_robot.move_to_joint_pos(joint_config)
                    print("TSR VALID: {}".format(test_tsr.is_valid(xyz+list(quat2rpy(orientation)))))
                    print_info(joint_config)
                if key_h in keys and keys[key_h] & p.KEY_WAS_TRIGGERED:
                    ycurrent-=dist
                    xyz = [xcurrent, ycurrent, zcurrent]
                    joint_config = sawyer_robot.solve_inverse_kinematics(xyz, orientation)
                    sawyer_robot.move_to_joint_pos(joint_config)
                    print("TSR VALID: {}".format(test_tsr.is_valid(xyz+list(quat2rpy(orientation)))))
                    print_info(joint_config)
                if key_j in keys and keys[key_j] & p.KEY_WAS_TRIGGERED:
                    xcurrent+=dist
                    xyz = [xcurrent, ycurrent, zcurrent]
                    joint_config = sawyer_robot.solve_inverse_kinematics(xyz, orientation)
                    sawyer_robot.move_to_joint_pos(joint_config)
                    print("TSR VALID: {}".format(test_tsr.is_valid(xyz+list(quat2rpy(orientation)))))
                    print_info(joint_config)
                if key_o in keys and keys[key_o] & p.KEY_WAS_TRIGGERED:
                    zcurrent+=dist
                    xyz = [xcurrent, ycurrent, zcurrent]
                    joint_config = sawyer_robot.solve_inverse_kinematics(xyz, orientation)
                    sawyer_robot.move_to_joint_pos(joint_config)
                    print("TSR VALID: {}".format(test_tsr.is_valid(xyz+list(quat2rpy(orientation)))))
                    print_info(joint_config)
                if key_l in keys and keys[key_l] & p.KEY_WAS_TRIGGERED:
                    zcurrent-=dist
                    xyz = [xcurrent, ycurrent, zcurrent]
                    joint_config = sawyer_robot.solve_inverse_kinematics(xyz, orientation)
                    sawyer_robot.move_to_joint_pos(joint_config)
                    print("TSR VALID: {}".format(test_tsr.is_valid(xyz+list(quat2rpy(orientation)))))
                    print_info(joint_config)
                if key_d in keys and keys[key_d] & p.KEY_WAS_TRIGGERED:
                    set_distance()
                if key_q in keys and keys[key_q] & p.KEY_WAS_TRIGGERED:
                    break
                

        


if __name__ == "__main__":
    main()
