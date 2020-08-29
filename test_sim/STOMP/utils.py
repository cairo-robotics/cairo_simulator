import json
import sys
import time

import pybullet as p

def load_configuration(file_name = "sawyer_configuration.json"):
    with open(file_name) as f:
        data = json.load(f)
        return data

def save_config_to_configuration_file(config, file_name = "sawyer_configuration.json"):
    with open(file_name, "r") as f:
        data = json.load(f)

    current_configs = data["sample_configurations"].keys()
    latest_config_number = max([int(key[6:]) for key in current_configs])
    data["sample_configurations"]['config' + str(latest_config_number+1)] = config

    with open(file_name, "w") as f:
        json.dump(data, f, indent=4, sort_keys=True)

# Function to manually control and save the desired state of the robot to the configuration file
def manual_control(robot, file_name = "sawyer_configuration.json"):
    concerned_arm_joints = robot._arm_dof_names
    arm_joint_limits = robot._arm_joint_limits
    current_state = robot.get_current_joint_states()

    save_pose_id = p.addUserDebugParameter("Save Pose", 1, 0, 1)
    previous_save_pose_value = p.readUserDebugParameter(save_pose_id)
    exit_id = p.addUserDebugParameter("Exit", 1, 0, 1)
    joint_ids = []
    for i in range(len(concerned_arm_joints)):
        joint_ids.append(p.addUserDebugParameter(concerned_arm_joints[i], arm_joint_limits[i][0],
                                arm_joint_limits[i][1], current_state[i]))

    while True:
        target_pose = []
        for joint_id in joint_ids:
            target_pose.append(p.readUserDebugParameter(joint_id))
        exit_value = p.readUserDebugParameter(exit_id)
        save_pose_value = p.readUserDebugParameter(save_pose_id)
        if exit_value == 2.0:
            break
        if save_pose_value > previous_save_pose_value:
            previous_save_pose_value = save_pose_value
            save_config_to_configuration_file(target_pose, file_name)
        robot.move_to_joint_pos(target_pose)
        time.sleep(0.01)
    p.disconnect()
    sys.exit(0)

# Method to add collision constraint for planning a trajectory
# Function borrowed from https://github.com/k-maheshkumar/trajopt_reimpl/blob/master/scripts/simulation/SimulationWorld.py
def create_cuboid_obstacle(name, shape , mass, position, orientation=None,
                      size=None, radius=None, height=None):

    if orientation is None:
        orientation = [0, 0, 0, 1]
    if position is not None:
        if radius is not None:
            col_id = p.createCollisionShape(shape, radius=radius, height=height)
            vis_id = p.createCollisionShape(shape, radius=radius, height=height)
        if size is not None:
            col_id = p.createCollisionShape(shape, halfExtents=size)
            vis_id = p.createCollisionShape(shape, halfExtents=size)

    shape_id = p.createMultiBody(mass, col_id, vis_id, basePosition=position, baseOrientation=orientation)
    return shape_id

######################################################################
### Code snippet to generate the Sawyer configuration json file ######
######################################################################

# excluded_pairs = [
#         (get_joint_info_by_name(sawyer_id, "right_l1").idx, get_joint_info_by_name(sawyer_id, "right_l0").idx),
#         (get_joint_info_by_name(sawyer_id, "right_l1").idx, get_joint_info_by_name(sawyer_id, "head").idx)]
#     sawyer_joint_limits_list = [['right_l0', (-3.0503, 3.0503)],
#                        ['right_l1', (-3.8095, 2.2736)],
#                        ['right_l2', (-3.0426, 3.0426)],
#                        ['right_l3', (-3.0439, 3.0439)],
#                        ['right_l4', (-2.9761, 2.9761)],
#                        ['right_l5', (-2.9761, 2.9761)],
#                        ['right_l6', (-4.7124, 4.7124)],
#                        ['right_gripper_l_finger', (0.0, 0.020833)],
#                        ['right_gripper_r_finger',
#                         (-0.020833, 0.0)],
#                        ['head', (-5.0952, 0.9064)]]
#
#     link_name_to_index = {'base': -1, 'torso': 0, 'pedestal': 1, 'right_arm_base_link': 2, 'right_l0': 3, 'head': 4, 'screen': 5, 'head_camera': 6, 'right_torso_itb': 7, 'right_l1': 8, 'right_l2': 9, 'right_l3': 10, 'right_l4': 11, 'right_arm_itb': 12, 'right_l5': 13, 'right_hand_camera': 14, 'right_wrist': 15, 'right_l6': 16, 'right_hand': 17, 'right_gripper_base': 18, 'right_connector_plate_base': 19, 'right_connector_plate_mount': 20, 'right_electric_gripper_base': 21, 'right_gripper_l_finger': 22, 'right_gripper_l_finger_tip': 23, 'right_gripper_r_finger': 24, 'right_gripper_r_finger_tip': 25, 'right_gripper_tip': 26}
#     sawyer_joint_limits = {link_name_to_index[row[0]]: row[1] for row in sawyer_joint_limits_list}
#     link_index_to_name = {link_name_to_index[name]:name for name in link_name_to_index}
#     sawyer_configuration = {'excluded_pairs': excluded_pairs,
#                             'sample_configurations': {'config0': list(start_state_config),
#                                                       'config1': list(goal_state_config)},
#                             'joint_limits': sawyer_joint_limits, 'link_index_to_name': link_index_to_name}
#                             # , 'link_pairs': link_pairs}
#
#     import json
#     with open('sawyer_configuration.json', 'w') as f:
#         json.dump(sawyer_configuration, f, indent=4, sort_keys=True)