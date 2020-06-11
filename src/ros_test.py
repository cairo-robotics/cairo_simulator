import time
import os
if os.environ.get('ROS_DISTRO'):
    import rospy

import pybullet as p

from cairo_simulator.simulator import Simulator, SimObject
from cairo_simulator.manipulators import Sawyer
from cairo_simulator.utils import ASSETS_PATH
from cairo_simulator.log import Logger


def main():
    if os.environ.get('ROS_DISTRO'):
        rospy.init_node("CAIRO_Sawyer_Simulator")
        use_ros = True
        logger = Logger(handlers=['ros'], level='debug')
        logger.info("INFO")
        logger.debug("DEBUG")
        logger.warn("WARN")
        logger.err("ERROR")
        logger.crit("CRIT")
    else:
        use_ros = False
    use_real_time = False

    sim = Simulator(logger=logger, use_ros=use_ros, use_real_time=use_real_time) # Initialize the Simulator
    ground_plane = SimObject("Ground", "plane.urdf", [0,0,0])
    sawyer_robot = Sawyer("sawyer0", [0, 0, 1.0])

    sim.publish_object_states()
    sim.publish_robot_states()


    # sim_obj = SimObject('cube0', 'cube_small.urdf', (0.75, 0, .55))
    # sim_obj = SimObject('cube1', 'cube_small.urdf', (0.74, 0.05, .55))
    # sim_obj = SimObject('cube2', 'cube_small.urdf', (0.67, -0.1, .55))
    # sim_obj = SimObject('cube3', 'cube_small.urdf', (0.69, 0.1, .55))

    # joint_config = sawyer_robot.solve_inverse_kinematics([0.9,0,1.5], [0,0,0,1])
    # #sawyer_robot.move_to_joint_pos(joint_config)

    # Loop until someone shuts us down
    # while rospy.is_shutdown() is not True:
    #     sim.step()
    time.sleep(5)
    p.disconnect()


if __name__ == "__main__":
    main()
