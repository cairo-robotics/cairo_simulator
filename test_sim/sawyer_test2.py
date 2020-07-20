import sys
import os
import time
import copy

if os.environ.get('ROS_DISTRO'):
    import rospy
import pybullet as p

from cairo_simulator.core.utils import ASSETS_PATH
from cairo_simulator.core.context import SawyerSimContext


def main():
    '''
    # Non-"configuration" way to initialize this demo
    use_ros = False
    use_real_time = True
    logger = Logger()
    sim = Simulator(logger=logger, use_ros=use_ros, use_real_time=use_real_time) # Initialize the Simulator
    ground_plane = SimObject("Ground", "plane.urdf", [0,0,0])

    # Add a table and a Sawyer robot
    table = SimObject("Table", ASSETS_PATH + 'table.sdf', (0.9, 0, 0), (0, 0, 1.5708)) # Table rotated 90deg along z-axis
    sawyer_robot = Sawyer("sawyer0", [0, 0, .8], fixed_base=1)

    sim_obj = SimObject('cube0', 'cube_small.urdf', (0.75, 0, .55))
    sim_obj = SimObject('cube1', 'cube_small.urdf', (0.74, 0.05, .55))
    sim_obj = SimObject('cube2', 'cube_small.urdf', (0.67, -0.1, .55))
    sim_obj = SimObject('cube3', 'cube_small.urdf', (0.69, 0.1, .55))
    '''

    configuration = {}

    configuration["sim"] = {
            "use_real_time": True
        }

    configuration["logging"] = {
            "handlers": ['logging'],
            "level": "debug"
        }

    configuration["sawyer"] = {
            "robot_name": "sawyer0",
            "position": [0, 0, 0.9],
            "fixed_base": True
        }

    configuration["sim_objects"] = [
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
        {
            "object_name": "cube0",
            "model_file_or_sim_id": "cube_small.urdf",
            "position": [0.75, 0, .55]
        },
        {
            "object_name": "cube1",
            "model_file_or_sim_id": "cube_small.urdf",
            "position": [0.74, 0.05, .55]
        },
        {
            "object_name": "cube2",
            "model_file_or_sim_id": "cube_small.urdf",
            "position": [0.67, -0.1, .55]
        },
        {
            "object_name": "cube3",
            "model_file_or_sim_id": "cube_small.urdf",
            "position": [0.69, 0.1, .55]
        }
    ]
    sim_context = SawyerSimContext(configuration)
    sim = sim_context.get_sim_instance()
    _ = sim_context.get_logger()
    _ = sim_context.get_state_space()
    _ = sim_context.get_state_validity()
    sawyer_robot = sim_context.get_robot()
    _ = sawyer_robot.get_simulator_id()
    _ = sim_context.get_sim_objects(['Ground'])[0]

    start_pos = [0]*7
    sawyer_robot.move_to_joint_pos(start_pos)

    joint_config = sawyer_robot.solve_inverse_kinematics(
        [0.9, 0, 1.5], [0, 0, 0, 1])

    joint_config2 = sawyer_robot.solve_inverse_kinematics(
        [0.7,0,1.5], [0,0,0,1])

    sawyer_robot.set_default_joint_velocity_pct(0.5)
    traj = ((1., joint_config), (2., joint_config2),
            (2.5, joint_config), (5, joint_config2))

    time.sleep(3)
    while sawyer_robot.check_if_at_position(start_pos, 0.5) is False:
        time.sleep(0.1)
        pass

    sawyer_robot.execute_trajectory(traj)

    # Loop until someone shuts us down
    try:
        while True:
            sim.step()
    except KeyboardInterrupt:
        p.disconnect()
        sys.exit(0)


if __name__ == "__main__":
    main()
