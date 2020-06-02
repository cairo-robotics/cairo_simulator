import pybullet as p
import rospy
from cairo_simulator.simulator import Simulator, SimObject
from cairo_simulator.manipulators import Sawyer
from planning.collision import self_collision_test, DisabledCollisionsContext
from cairo_simulator.sim_const import ASSETS_PATH
from cairo_motion_planning.samplers import UniformSampler
from cairo_motion_planning.state_space import SawyerConfigurationSpace
from cairo_motion_planning.state_validity import StateValidyChecker

from functools import partial

import timeit

def main():
    rospy.init_node("CAIRO_Sawyer_Simulator")
    use_real_time = True

    sim = Simulator(gui=True) # Initialize the Simulator

    # Add a table and a Sawyer robot
    table = SimObject("Table", ASSETS_PATH + 'table.sdf', (0.9, 0, 0), (0, 0, 1.5708)) # Table rotated 90deg along z-axis
    sawyer_robot = Sawyer("sawyer0", 0, 0, 0.8)

    self_collision_fn = partial(self_collision_test, robot=sawyer_robot, excluded_pairs=[(5, 33), (6, 33)])

    svc = StateValidyChecker(self_collision_fn)
    scs = SawyerConfigurationSpace()
    sampler = UniformSampler(scs.get_bounds())

    n_samples = 1000
    valid_samples = []
    starttime = timeit.default_timer()
    with DisabledCollisionsContext(sim):
        print("Sampling start time is :",starttime)
        for i in range(0, n_samples):
            sample = sampler.sample()
            if svc.validate(sample):
                valid_samples.append(sample)
        print("The time difference is :", timeit.default_timer() - starttime)
        print("{} valid of {}".format(len(valid_samples), n_samples))

    # Loop until someone shuts us down
    # while rospy.is_shutdown() is not True:
    #     sim.step()
    p.disconnect()


if __name__ == "__main__":
    main()