import time
import os
if os.environ.get('ROS_DISTRO'):
    import rospy

import pybullet as p

from cairo_simulator.core.simulator import Simulator, SimObject
from cairo_simulator.core.log import Logger
from cairo_simulator.devices.manipulators import Sawyer


def main():
    """
    Illustrates the difference between running within a ROS environment, and running the vanilla Simulator.
    In sourcing your ROS environment (e.g. source /opt/ros/melodic/setup.bash) certain ROS specific environment variables
    are set, such as ROS_DISTRO. This could be used to pivot between using ROS and not using ROS. This is really
    for illustrative purposes mostly. 

    For your own scripts, you could simply write them under the assumption that you will either be using ROS or not, 
    and the user of your script should be aware of this fact or may run into issues running in either environment.
    """
    if os.environ.get('ROS_DISTRO'):
        # The Logger class is a wrapper around Python's own logging, print(), and rospy logging. See cairo_simulator.log.Logger
        logger = Logger(handlers=['ros'], level='info')
        # Awkwardly, the logging level for ROS must be set in the init_node function call. I do not see a workaround as of right now.
        rospy.init_node("CAIRO_Sawyer_Simulator", log_level=rospy.DEBUG)
        # Example of a logging statement. There is also .debug() .warn() .crit() .err()
        logger.info("INFO")

        # If using ROS, make sure to set the use_ros flag in the simulator.
        use_ros = True
        use_real_time = False

        # Initialize the Simulator by passing in the logger to be used internally and any appropraite flags.
        sim = Simulator(logger=logger, use_ros=use_ros, use_real_time=use_real_time)
        ground_plane = SimObject("Ground", "plane.urdf", [0,0,0])
        sawyer_robot = Sawyer("sawyer0", [0, 0, 1.0])

        # Try calling ROS specific methods within the Simulator. 
        try:
            logger.info("Attempting to execute ROS-based methods.")
            sim.publish_object_states()
            sim.publish_robot_states()
            logger.info("Successfully executed ROS-based methods in the Simulator.")
        except Exception as e:
            logger.info("Failed to execute ROS methods within the Simulator.")
            logger.info(e)
        time.sleep(1)
        p.disconnect()
    else:
        # If running this script without sourcing any ROS environment, this else block section should execute.
        # Notice we've indicated Python's logger in our Logger class.
        logger = Logger(handlers=['logging'], level='info')
        use_ros = False
        use_real_time = False
        sim = Simulator(logger=logger, use_ros=use_ros, use_real_time=use_real_time) # Initialize the Simulator
        ground_plane = SimObject("Ground", "plane.urdf", [0,0,0])
        sawyer_robot = Sawyer("sawyer0", [0, 0, 1.0])

        # This methods should indeed fail and the except block should be reached.
        try:
            logger.info("Attempting to execute ROS-based methods.")
            sim.publish_object_states()
            sim.publish_robot_states()
        except Exception as e:
            logger.info("Failed to execute ROS-based methods within the Simulator.")
            logger.info(e)

        time.sleep(1)
        p.disconnect()

if __name__ == "__main__":
    main()
