import time
import os
import sys
from pathlib import Path
import numpy as np

if os.environ.get('ROS_DISTRO'):
    import rospy
import pybullet as p

from cairo_simulator.core.utils import ASSETS_PATH
from cairo_simulator.core.log import Logger
from cairo_simulator.core.simulator import Simulator, SimObject
from cairo_simulator.devices.manipulators import Sawyer
from cairo_planning.geometric.transformation import pose2trans, pseudoinverse, analytic_xyz_jacobian, quat2rpy, rpy2quat
from cairo_planning.geometric.utils import angle_between
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
    ground_plane = SimObject("Ground", "plane.urdf", [0, 0, -.8])
    # Add a table and a Sawyer robot
    # table = SimObject("Table", ASSETS_PATH + 'table.sdf', (0.9, 0, 0),
    #                   (0, 0, 1.5708))  # Table rotated 90deg along z-axis
    # print(p.getNumJoints(table.get_simulator_id()))
    sawyer_robot = Sawyer("sawyer0", [0, 0, 0.0], fixed_base=1)
    settings_path = str(Path(__file__).parent.absolute()) + "/settings.yaml"
    rusty_sawyer_robot = Agent(settings_path, False, False)
 
    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)

    # sim_obj = SimObject('cube0', 'cube_small.urdf', (0.75, 0, .55))
    # sim_obj = SimObject('cube1', 'cube_small.urdf', (0.74, 0.05, .55))
    # sim_obj = SimObject('cube2', 'cube_small.urdf', (0.67, -0.1, .55))
    # sim_obj = SimObject('cube3', 'cube_small.urdf', (0.69, 0.1, .55))
    test_fk_config = [ 0.0, 0.0, -1.5708, 1.5708, 0.0, -1.5708, 0.0 ]
    # sawyer_robot.move_to_joint_pos(joint_config)
    reg_fk = sawyer_robot.solve_forward_kinematics(test_fk_config)[0]
    
    print(reg_fk[0], quat2rpy(reg_fk[1]))

    
    rusty_fk = rusty_sawyer_robot.forward_kinematics(test_fk_config)
    rust_euler = quat2rpy(rusty_fk[1])
    # rust_euler[-1] = rust_euler[-1] + np.pi/2
    print(rusty_fk[0], rust_euler)


if __name__ == "__main__":
    main()
