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
    logger = Logger(handlers=['std', 'logging'], level='debug')
    logger.info("INFO")
    logger.debug("DEBUG")
    logger.warn("WARN")
    logger.err("ERROR")
    logger.crit("CRIT")
    # use_real_time = True

    # sim = Simulator() # Initialize the Simulator
    # ground_plane = SimObject("Ground", "plane.urdf", [0,0,0])
    # sawyer_robot = Sawyer("sawyer0", [0, 0, 0.8])

    # # Add a table and a Sawyer robot
    # table = SimObject("Table", ASSETS_PATH + 'table.sdf', (0.9, 0, 0), (0, 0, 1.5708)) # Table rotated 90deg along z-axis
    # print(p.getNumJoints(table.get_simulator_id()))

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
