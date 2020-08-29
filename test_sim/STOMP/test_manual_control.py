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
from utils import load_configuration, save_config_to_configuration_file, manual_control, create_cuboid_obstacle

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
    # sim_obj1 = SimObject('cube0', 'cube_small.urdf', (0.75, 0, .55))
    # sim_obj2 = SimObject('cube1', 'cube_small.urdf', (0.74, 0.05, .55))
    # sim_obj3 = SimObject('cube2', 'cube_small.urdf', (0.67, -0.1, .55))
    # sim_obj4 = SimObject('cube3', 'cube_small.urdf', (0.69, 0.1, .55))

    sim_obj1 = create_cuboid_obstacle(name='box0', shape=p.GEOM_BOX, mass=1, position=[0.74, 0.05, .55],
                                      size=[0.09, 0.09, 0.35])
    # Interested in only PyBullet object IDs for obstacles
    obstacles = [sim_obj1, table.get_simulator_id()]

    return sim, sawyer_robot, obstacles

def main():
    # Load sawyer specific configuration
    sawyer_configuration = load_configuration('sawyer_configuration.json')
    sample_configurations = sawyer_configuration['sample_configurations']

    # Setup simulator
    sim, sawyer_robot, obstacles = init_sim_with_sawyer()

    # Moving Sawyer to a start position
    sawyer_robot.move_to_joint_pos(sample_configurations['config0'])
    time.sleep(2)
    manual_control(sawyer_robot)

if __name__ == "__main__":
    main()