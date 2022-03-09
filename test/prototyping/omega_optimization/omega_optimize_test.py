import time
import os
from pathlib import Path
import sys
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
    print(settings_path)
    rusty_sawyer_robot = Agent(settings_path, False, False)

    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
    # [0.7511185307142259, -0.20750105802527086, 0.3492014692886301] [ 1.57079633 -1.39626233  1.57078898]

    # joint_config_relaxed_ik = rusty_sawyer_robot.omega_projection([.2,0,-.5], [0,0,0,1]).data
    # print(joint_config_relaxed_ik)
    tsr_config = {
        'degrees': False,
        "T0_w":  [.7968, -.5772, 0.15, np.pi/2, 0,  np.pi/2],
        "Tw_e": [0, 0, 0, 0, 0, 0],
        "Bw": [[(-.05, .05), (-.05, .05), (-100, 100)],  
                [(-.05, .05), (-.05, .05), (-.05, .05)]]
    }
        

    # T0_w = [0, 0, 0, 0, 0, 0]
    # Tw_e = [.5, 0, 0, np.pi/2,  0,  np.pi/2]
    # Bw_np = np.zeros((6, 2))
    # Bw_np[0, :] = [-5, 5]
    # Bw_np[1, :] = [0, 0]
    # Bw_np[2, :] = [0, 0]  # Allow a little vertical movement
    # Bw_np[3, :] = [-.03, .03]
    # Bw_np[4, :] = [-.03, .03]
    # Bw_np[5, :] = [-.03, .03]
    # Bw = [list(bounds) for bounds in Bw_np]
    sample = [-1.0541111350003591, 0.47938957018657513, -1.2804332159469978, 0.1631097390425542, -0.0701496186869246, -1.2661493531546648, -0.3275759417945361]
    # sample = [0, 0, 0, 0, 0, 0, 0]
    sawyer_robot.set_joint_state(sample)
    time.sleep(5)
    seed_start = sawyer_robot.solve_inverse_kinematics(tsr_config["T0_w"][0:3], tsr_config["T0_w"][3:])
    rusty_sawyer_robot.update_xopt(seed_start)
    rusty_sawyer_robot.update_tsr(tsr_config["T0_w"], tsr_config["Tw_e"], tsr_config["Bw"][0] + tsr_config["Bw"][1])
    try:
        while True:    
            joint_config_relaxed_ik = rusty_sawyer_robot.omega_optimize(sample).data
            sawyer_robot.set_joint_state(joint_config_relaxed_ik)
    except KeyboardInterrupt:
        p.disconnect()
        sys.exit(0)


if __name__ == "__main__":
    main()
