import os

import pybullet as p
if os.environ.get('ROS_DISTRO'):
    import rospy

from cairo_simulator.simulator import Simulator, SimObject
from cairo_simulator.manipulators import Sawyer
from cairo_simulator.log import Logger
from planning.collision import self_collision_test, DisabledCollisionsContext
from cairo_simulator.utils import ASSETS_PATH
from cairo_simulator.link import get_joint_info_by_name

from cairo_motion_planning.samplers import UniformSampler
from cairo_motion_planning.state_space import SawyerConfigurationSpace
from cairo_motion_planning.state_validity import StateValidyChecker


from functools import partial

import timeit

def main():
    if os.environ.get('ROS_DISTRO'):
        rospy.init_node("CAIRO_Sawyer_Simulator")
        use_ros = True
    else:
        use_ros = False
    use_real_time = True
    logger = Logger()
    sim = Simulator(logger=logger, use_ros=use_ros) # Initialize the Simulator

    ground_plane = SimObject("Ground", "plane.urdf", [0,0,0])


    # Add a table and a Sawyer robot
    table = SimObject("Table", ASSETS_PATH + 'table.sdf', (0.9, 0, 0), (0, 0, 1.5708)) # Table rotated 90deg along z-axis
    sawyer_robot = Sawyer("sawyer0", [0, 0, 0.9])

    self_collision_fn = partial(self_collision_test, robot=sawyer_robot, excluded_pairs=[(4, 30), (5, 30)])

    svc = StateValidyChecker(self_collision_fn)
    scs = SawyerConfigurationSpace()
    sampler = UniformSampler(scs.get_bounds())

    n_samples = 1000
    valid_samples = []
    starttime = timeit.default_timer()

    # Exclude the ground plane and the pedestal feet from disabled collisions.
    excluded_bodies = [ground_plane.get_simulator_id()] # the ground plane
    sawyer_body_idx = sawyer_robot.get_simulator_id()
    pedestal_feet_idx = get_joint_info_by_name(sawyer_robot.get_simulator_id(), 'pedestal_feet').idx
    body_link_pair = (sawyer_body_idx, pedestal_feet_idx) # The (sawyer_idx, pedestal_feet_idx) tuple the ecluded from disabled collisions.
    excluded_body_link_pairs = [body_link_pair]

    with DisabledCollisionsContext(sim, excluded_bodies, excluded_body_link_pairs):
        print("Sampling start time is :",starttime)
        for i in range(0, n_samples):
            sample = sampler.sample()
            if svc.validate(sample):
                valid_samples.append(sample)
        print("The time difference is :", timeit.default_timer() - starttime)
        print("{} valid of {}".format(len(valid_samples), n_samples))
    # print("Sampling start time is :",starttime)
    # for i in range(0, n_samples):
    #     sample = sampler.sample()
    #     if svc.validate(sample):
    #         valid_samples.append(sample)
    # print("The time difference is :", timeit.default_timer() - starttime)
    # print("{} valid of {}".format(len(valid_samples), n_samples))

    # Loop until someone shuts us down
    # while rospy.is_shutdown() is not True:
    #     sim.step()
    p.disconnect()


if __name__ == "__main__":
    main()

    import time
import os
import sys

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
    use_real_time = True
    logger = Logger()
    sim = Simulator(logger=logger, use_ros=use_ros) # Initialize the Simulator
    ground_plane = SimObject("Ground", "plane.urdf", [0,0,0])
    # Add a table and a Sawyer robot
    table = SimObject("Table", ASSETS_PATH + 'table.sdf', (0.9, 0, 0), (0, 0, 1.5708)) # Table rotated 90deg along z-axis
    print(p.getNumJoints(table.get_simulator_id()))
    sawyer_robot = Sawyer("sawyer0", [0, 0, 0.8])

    sim_obj = SimObject('cube0', 'cube_small.urdf', (0.75, 0, .55))
    sim_obj = SimObject('cube1', 'cube_small.urdf', (0.74, 0.05, .55))
    sim_obj = SimObject('cube2', 'cube_small.urdf', (0.67, -0.1, .55))
    sim_obj = SimObject('cube3', 'cube_small.urdf', (0.69, 0.1, .55))

    joint_config = sawyer_robot.solve_inverse_kinematics([0.9,0,1.5], [0,0,0,1])
    #sawyer_robot.move_to_joint_pos(joint_config)

    # Loop until someone shuts us down
    try:
        while True:
            sim.step()
    except KeyboardInterrupt:
        p.disconnect()
        sys.exit(0)
   



if __name__ == "__main__":
    main()
