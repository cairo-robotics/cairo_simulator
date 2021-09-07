import os
import json
import time
from functools import partial
import datetime

import pybullet as p
if os.environ.get('ROS_DISTRO'):
    import rospy
import numpy as np

from cairo_simulator.core.sim_context import SawyerBiasedCPRMSimContext
from cairo_simulator.core.utils import ASSETS_PATH

from cairo_planning.collisions import DisabledCollisionsContext
from cairo_planning.local.interpolation import parametric_lerp
from cairo_planning.local.curve import JointTrajectoryCurve
from cairo_planning.planners import CPRM
from cairo_planning.sampling.samplers import HyperballSampler
from cairo_planning.geometric.state_space import SawyerConfigurationSpace
from cairo_planning.core.serialization import load_model


def main():
    # Reload the samples and configuration
    directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data/serialization_data/test_model")
    config, samples, graph = load_model(directory)

    config['tsr'] = {
            'degrees': False,
            "T0_w": [.7, 0, 0, 0, 0, 0],
            "Tw_e": [-.2, 0, .739, -3.1261701132911655, 0.023551837572146628, 0.060331404738664496],
            "Bw": [[(0, 100), (-100, 100), (-.1, 0)],  # allow some tolerance in the z and y and only positve in x
                    [(-.07, .07), (-.07, .07), (-.07, .07)]]
        }

    start = [
        0.673578125,
        -0.2995908203125,
        -0.21482421875,
        1.4868740234375,
        0.53829296875,
        0.4117080078125,
        -1.2169501953125]

    goal = [
        -1.3020732421875,
        -0.44705859375,
        0.6508818359375,
        1.5064189453125,
        -0.889978515625,
        0.8245869140625,
        -1.6826474609375]


    # Collect all joint configurations from all demonstration .json files.
    configurations = []
    data_directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data/sampling_bias")
  
    print("Running biased sampling test for {}".format(data_directory))
    for json_file in os.listdir(data_directory):
        filename = os.path.join(data_directory, json_file)
        with open(filename, "r") as f:
            data = json.load(f)
            for entry in data:
                configurations.append(entry['robot']['joint_angle'])

    config['sampling_bias'] = {
        'bandwidth': .1,
        'fraction_uniform': .25,
        'data': configurations
    }

    sim_context = SawyerBiasedCPRMSimContext(configuration=config)
    sim = sim_context.get_sim_instance()
    logger = sim_context.get_logger()
    sawyer_robot = sim_context.get_robot()
    biased_state_space = sim_context.get_state_space()
    svc = sim_context.get_state_validity()
    tsr = sim_context.get_tsr()
    sawyer_robot.move_to_joint_pos(start)
    sawyer_robot.move_to_joint_pos(start)
    time.sleep(2)
    # Utilizes RPY convention
    with DisabledCollisionsContext(sim, [], []):
        ###########
        # LazyPRM #
        ###########
        # The specific space we sample from is the Hyberball centered at the midpoint between two candidate points. 
        # This is used to bias tree grwoth between two points when using CBiRRT2 as our local planner for a constrained PRM.
        tree_state_space = SawyerConfigurationSpace(sampler=HyperballSampler())
        # Use parametric linear interpolation with 10 steps between points.
        interp = partial(parametric_lerp, steps=10)
        # See params for PRM specific parameters
        prm = CPRM(SawyerBiasedCPRMSimContext, config, sawyer_robot, tsr, biased_state_space, tree_state_space, svc, interp, params={
            'n_samples': 3000, 'k': 8, 'planning_attempts': 5, 'ball_radius': 2.0, 'smooth_path': True, 'smoothing_time':10}, tree_params={'iters': 50, 'q_step': .5}, logger=logger)
        logger.info("Planning....")
        prm.preload(samples, graph)
        path = prm.plan(np.array(start), np.array(goal))
    # splining uses numpy so needs to be converted
    path = [np.array(p) for p in path]
    # Create a MinJerk spline trajectory using JointTrajectoryCurve and execute
    jtc = JointTrajectoryCurve()
    traj = jtc.generate_trajectory(path, move_time=20)
    sawyer_robot.execute_trajectory(traj)
    try:
        while True:
            sim.step()
    except KeyboardInterrupt:
        p.disconnect()
        sys.exit(0)
   

if __name__ == "__main__":
    main()
