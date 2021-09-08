import sys
import os
import time
from cairo_planning.collisions import DisabledCollisionsContext

if os.environ.get('ROS_DISTRO'):
    import rospy
import pybullet as p

from cairo_simulator.core.utils import ASSETS_PATH
from cairo_simulator.core.sim_context import SawyerSimContext


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
            "use_real_time": False
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
        # {
        #     "object_name": "Table",
        #     "model_file_or_sim_id": ASSETS_PATH + 'table.sdf',
        #     "position": [0.9, 0, 0],
        #     "orientation":  [0, 0, 1.5708]
        # },
        # {
        #     "object_name": "cube0",
        #     "model_file_or_sim_id": "cube_small.urdf",
        #     "position": [0.75, 0, .55]
        # },
        # {
        #     "object_name": "cube1",
        #     "model_file_or_sim_id": "cube_small.urdf",
        #     "position": [0.74, 0.05, .55]
        # },
        # {
        #     "object_name": "cube2",
        #     "model_file_or_sim_id": "cube_small.urdf",
        #     "position": [0.67, -0.1, .55]
        # },
        # {
        #     "object_name": "cube3",
        #     "model_file_or_sim_id": "cube_small.urdf",
        #     "position": [0.69, 0.1, .55]
        # }
    ]
    configuration["primitives"] = [
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
    sim_context = SawyerSimContext(configuration)
    sim = sim_context.get_sim_instance()
    _ = sim_context.get_logger()
    _ = sim_context.get_state_space()
    svc = sim_context.get_state_validity()
    sawyer_robot = sim_context.get_robot()
    _ = sawyer_robot.get_simulator_id()
    _ = sim_context.get_sim_objects(['Ground'])[0]
    with DisabledCollisionsContext(sim, [], [], disable_visualization=False):
        start_pos = [0]*7
        # sawyer_robot.move_to_joint_pos(start_pos)

        joint_config = sawyer_robot.solve_inverse_kinematics(
            [ 0.7074708676269519,  -0.08765564452576573, 0.9], [0, 0, 0, 1])

        # joint_config2 = sawyer_robot.solve_inverse_kinematics(
        #     [ 0.7074708676269519, -0.08765564452576573, 0.744752279781234485], 
        #     [-0.9811850913657468, -0.0013224117851769242, -0.19302238661442592, 0.0040528970296797445])

        # joint_config2 = sawyer_robot.solve_inverse_kinematics( [ 0.7074708676269519,  -0.08765564452576573, 0.9], [0, 0, 0, 1])

        joint_config2 = [-0.93220603, -0.38925626,  0.22872595,  1.47503778, -0.60152948,
            0.459639  , -1.87886531]
        sawyer_robot.set_default_joint_velocity_pct(0.5)
        traj = ((1., joint_config), (2, joint_config2))

        while sawyer_robot.check_if_at_position(start_pos, 0.5) is False:
            time.sleep(0.1)
            pass
        print(svc.validate(joint_config2))
        sawyer_robot.execute_trajectory(traj)
    
        # Loop until someone shuts us down
        try:
            while True:
                sim.step()
                print(sawyer_robot.get_current_joint_states())
        except KeyboardInterrupt:
            p.disconnect()
            sys.exit(0)


if __name__ == "__main__":
    main()
