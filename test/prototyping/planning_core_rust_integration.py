import time
import os
import sys

if os.environ.get('ROS_DISTRO'):
    import rospy
import pybullet as p

from cairo_simulator.core.utils import ASSETS_PATH
from cairo_simulator.core.log import Logger
from cairo_simulator.core.simulator import Simulator, SimObject
from cairo_simulator.devices.manipulators import Sawyer

from cairo_planning_core import Agent


def main():
    if os.environ.get('ROS_DISTRO'):
        rospy.init_node("CAIRO_Sawyer_Simulator")
        use_ros = True
    else:
        use_ros = False
    use_real_time = True
    logger = Logger()
    sim = Simulator(logger=logger, use_ros=use_ros,
                    use_real_time=use_real_time)  # Initialize the Simulator
    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
    p.setPhysicsEngineParameter(enableFileCaching=0)
    ground_plane = SimObject("Ground", "plane.urdf", [0, 0, 0])
    # Add a table and a Sawyer robot
    # table = SimObject("Table", ASSETS_PATH + 'table.sdf', (0.9, 0, 0),
    #                   (0, 0, 1.5708))  # Table rotated 90deg along z-axis
    # print(p.getNumJoints(table.get_simulator_id()))
    sawyer_robot = Sawyer("sawyer0", [0, 0, 0.8], fixed_base=1)
    rusty_sawyer_robot = Agent(
        '/home/carl/cairo/cairo_simulator/test/prototyping/settings.yaml', False, False)

    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)

    # sim_obj = SimObject('cube0', 'cube_small.urdf', (0.75, 0, .55))
    # sim_obj = SimObject('cube1', 'cube_small.urdf', (0.74, 0.05, .55))
    # sim_obj = SimObject('cube2', 'cube_small.urdf', (0.67, -0.1, .55))
    # sim_obj = SimObject('cube3', 'cube_small.urdf', (0.69, 0.1, .55))

    # sawyer_robot.move_to_joint_pos(joint_config)
    print(sawyer_robot.solve_forward_kinematics([
        -1.3020732421875,
        -0.44705859375,
        0.6508818359375,
        1.5064189453125,
        -0.889978515625,
        0.8245869140625,
        -1.6826474609375]))
    
    print(rusty_sawyer_robot.forward_kinematics([
        -1.3020732421875,
        -0.44705859375,
        0.6508818359375,
        1.5064189453125,
        -0.889978515625,
        0.8245869140625,
        -1.6826474609375]))
    # Loop until someone shuts us down
    try:
        while True:
            # joint_config = sawyer_robot.solve_inverse_kinematics([1,0,0], [0,0,0,1])
            # sawyer_robot.set_joint_state(joint_config)
            # time.sleep(5)
            joint_config_relaxed_ik = rusty_sawyer_robot.relaxed_inverse_kinematics([1,0,-.8], [0,0,0,1]).data
            sawyer_robot.set_joint_state(joint_config_relaxed_ik)
    except KeyboardInterrupt:
        p.disconnect()
        sys.exit(0)


if __name__ == "__main__":
    main()
