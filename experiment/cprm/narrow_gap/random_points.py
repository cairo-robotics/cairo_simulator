import random
import time
import copy

from cairo_simulator.core.sim_context import SawyerBiasedCPRMSimContext
from cairo_simulator.core.utils import ASSETS_PATH


def main():

    config = {}
    config["sim"] = {
            "use_real_time": False
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

    start = [
        0.673578125,
        -0.2995908203125,
        -0.21482421875,
        1.4868740234375,
        0.53829296875,
        0.4117080078125,
        -1.2169501953125]

    goal = [
        -1.3020732421875,
        -0.44705859375,
        0.6508818359375,
        1.5064189453125,
        -0.889978515625,
        0.8245869140625,
        -1.6826474609375]

    sim_context = SawyerBiasedCPRMSimContext(configuration=config)
    sawyer_robot = sim_context.get_robot()
    svc = sim_context.get_state_validity()

    start_world_pose, start_local_pose = sawyer_robot.solve_forward_kinematics(
        start)
    goal_world_pose, goal_local_pose = sawyer_robot.solve_forward_kinematics(
        goal)

    # print(start_world_pose, start_local_pose)
    # print(sawyer_robot.solve_inverse_kinematics(start_world_pose[0], start_world_pose[1]))
    # exit()

    random_start_configurations = []
    for _ in range(0, 10):
        while True:
            delta_x = random.uniform(.2, .2)
            delta_y = random.uniform(-.2, .2)
            new_position = copy.deepcopy([copy.deepcopy(start_world_pose[0][0]) + delta_x, copy.deepcopy(start_world_pose[0][1]) + delta_y, copy.deepcopy(start_world_pose[0][2])])
            new_config = sawyer_robot.solve_inverse_kinematics(new_position, copy.deepcopy(start_world_pose[1]))
            if svc.validate(new_config):
                new_config
                break
        random_start_configurations.append(copy.deepcopy(new_config))
    
    random_goal_configurations = []
    for _ in range(0, 10):
        while True:
            delta_x = random.uniform(-.1, .1)
            delta_y = random.uniform(-.1, .1)
            new_position = [goal_world_pose[0][0] + delta_x, goal_world_pose[0][1] + delta_y, goal_world_pose[0][2]]
            new_config = sawyer_robot.solve_inverse_kinematics(new_position, goal_world_pose[1])
            if svc.validate(new_config):
                break
        random_goal_configurations.append(new_config)

    print(random_start_configurations)
    print(random_goal_configurations)

if __name__ == "__main__":
    main()
