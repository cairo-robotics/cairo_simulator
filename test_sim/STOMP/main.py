import time
import os
import sys
from pprint import pprint

import pybullet as p
import numpy as np

from cairo_simulator.core.utils import ASSETS_PATH
from cairo_simulator.core.log import Logger
from cairo_simulator.core.simulator import Simulator, SimObject
from cairo_simulator.devices.manipulators import Sawyer
from cairo_simulator.core.link import get_link_pairs, get_joint_info_by_name
from STOMP import STOMP
from utils import load_configuration, save_config_to_configuration_file



def init_sim_with_sawyer():
    # Setup simulator
    use_ros = False
    use_real_time = True
    logger = Logger()
    sim = Simulator(logger=logger, use_ros=use_ros, use_real_time=use_real_time)  # Initialize the Simulator
    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
    p.setPhysicsEngineParameter(enableFileCaching=0)
    ground_plane = SimObject("Ground", "plane.urdf", [0, 0, 0])
    # Add a table and a Sawyer robot
    print("Path = ", os.path.abspath(ASSETS_PATH))
    table = SimObject("Table", ASSETS_PATH + 'table.sdf', (0.9, 0, 0),
                      (0, 0, 1.5708))  # Table rotated 90deg along z-axis
    sawyer_robot = Sawyer("sawyer0", [0, 0, 0.8], fixed_base=1)

    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
    # Add small cubes on the table (table top bounds are known in world coordinate)
    # TODO add large obstacles
    sim_obj1 = SimObject('cube0', 'cube_small.urdf', (0.75, 0, .55))
    sim_obj2 = SimObject('cube1', 'cube_small.urdf', (0.74, 0.05, .55))
    sim_obj3 = SimObject('cube2', 'cube_small.urdf', (0.67, -0.1, .55))
    sim_obj4 = SimObject('cube3', 'cube_small.urdf', (0.69, 0.1, .55))
    obstacles = [sim_obj1, sim_obj2, sim_obj3, sim_obj4, table]
    return sim, sawyer_robot, obstacles

def main():
    # Load sawyer specific configuration
    sawyer_configuration = load_configuration('sawyer_configuration.json')
    excluded_pairs = sawyer_configuration['excluded_pairs']
    sample_configurations = sawyer_configuration['sample_configurations']

    # Setup simulator
    sim, sawyer_robot, obstacles = init_sim_with_sawyer()

    # Moving Sawyer to a start position
    sawyer_robot.move_to_joint_pos(sample_configurations['config0'])
    time.sleep(3)


    start_state_config = sample_configurations['config0']
    goal_state_config = sample_configurations['config1']
    sawyer_id = sawyer_robot.get_simulator_id()
    link_pairs = get_link_pairs(sawyer_id, excluded_pairs=excluded_pairs)

    # Initializing STOMP
    stomp = STOMP(sim, sawyer_robot, link_pairs, obstacles,
                  start_state_config, goal_state_config, N=10)


    # stomp.print_trajectory()
    #
    # stomp.plan()
    # stomp.print_trajectory()
    # trajectory_data = stomp.get_trajectory(1)
    # sawyer_robot.execute_trajectory(trajectory_data)

    # Loop until someone shuts us down
    try:
        while True:
            sim.step()
    except KeyboardInterrupt:
        p.disconnect()
        sys.exit(0)


if __name__ == "__main__":
    main()
