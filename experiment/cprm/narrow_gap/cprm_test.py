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
    directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data/biased_serialized_model")
    _, samples, graph = load_model(directory)
    config = {}
    config["sim"] = {
            "use_real_time": False
        }

    config["logging"] = {
            "handlers": ['logging'],
            "level": "debug"
        }

    config["sim_objects"] = [
        {
            "object_name": "Ground",
            "model_file_or_sim_id": "plane.urdf",
            "position": [0, 0, 0]
        },
        {
            "object_name": "Table",
            "model_file_or_sim_id": ASSETS_PATH + 'table.sdf',
            "position": [0.9, 0, 0],
            "orientation":  [0, 0, 1.5708]
        },
    ]
    config["primitives"] = [
        {
            "type": "box",
            "primitive_configs": {"w": .2, "l": .45, "h": .35},
            "sim_object_configs": 
                {
                    "object_name": "box",
                    "position": [.6, 0, .7],
                    "orientation":  [0, 0, 0],
                    "fixed_base": 1    
                }
        },
        {
            "type": "box",
            "primitive_configs": {"w": .2, "l": .45, "h": .35},
            "sim_object_configs": 
                {
                    "object_name": "box",
                    "position": [.9, 0, .7],
                    "orientation":  [0, 0, 0],
                    "fixed_base": 1    
                }
        },
        ]
            
    config['tsr'] = {
            'degrees': False,
            "T0_w": [.7, 0, 0, 0, 0, 0],
            "Tw_e": [-.2, 0, .739, -3.1261701132911655, 0.023551837572146628, 0.060331404738664496],
            "Bw": [[(0, 100), (-100, 100), (-.1, 0)],  
                    [(-.07, .07), (-.07, .07), (-.07, .07)]]
        }

    start = [0.8140604621711953, 0.5784497787000918, -1.3840668226493182, 0.32705959235984394, -1.6190593151010773, -0.9726661317216014, 2.192982202249367]

    goal = [-0.5208792405945148, 0.3497179558781994, -0.8790919912955788, 0.9982353480155052, -1.6491489611308177, -0.8044683949630524, 0.16088445579362776]

    # Collect all joint configurations from all demonstration .json files.
    # configurations = []
    # data_directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data/sampling_bias")
  
    # print("Running biased sampling test for {}".format(data_directory))
    # for json_file in os.listdir(data_directory):
    #     filename = os.path.join(data_directory, json_file)
    #     with open(filename, "r") as f:
    #         data = json.load(f)
    #         for entry in data:
    #             configurations.append(entry['robot']['joint_angle'])

    # config['sampling_bias'] = {
    #     'bandwidth': .2,
    #     'fraction_uniform': .50,
    #     'data': configurations
    # }

    sim_context = SawyerBiasedCPRMSimContext(configuration=config)
    sim = sim_context.get_sim_instance()
    logger = sim_context.get_logger()
    sawyer_robot = sim_context.get_robot()
    biased_state_space = sim_context.get_state_space()
    svc = sim_context.get_state_validity()
    tsr = sim_context.get_tsr()
    sawyer_robot.set_joint_state(start)
    time.sleep(3)
    # Utilizes RPY convention
    with DisabledCollisionsContext(sim, [], []):
        ###########
        # CPRM #
        ###########
        # The specific space we sample from is the Hyberball centered at the midpoint between two candidate points. 
        # This is used to bias tree grwoth between two points when using CBiRRT2 as our local planner for a constrained PRM.
        tree_state_space = SawyerConfigurationSpace(sampler=HyperballSampler())
        # Use parametric linear interpolation with 10 steps between points.
        interp = partial(parametric_lerp, steps=10)
        # See params for PRM specific parameters
        prm = CPRM(SawyerBiasedCPRMSimContext, config, sawyer_robot, tsr, biased_state_space, tree_state_space, svc, interp, params={
            'n_samples': 3000, 'k': 8, 'planning_attempts': 5, 'ball_radius': 2.0, 'smooth_path': True, 'smoothing_time':10}, tree_params={'iters': 100, 'q_step': .1}, logger=logger)
        logger.info("Preloading samples and model....")
        prm.preload(samples, graph)
        logger.info("Planning.")
        path = prm.plan(np.array(start), np.array(goal))
    # splining uses numpy so needs to be converted
    path = np.array([np.array(p) for p in path])
    # Create a MinJerk spline trajectory using JointTrajectoryCurve and execute
    jtc = JointTrajectoryCurve()
    traj = jtc.generate_trajectory(path, move_time=10)
    input("Press any key to execute...")
    try:
        prior_time = 0
        for i, point in enumerate(traj):
            if not svc.validate(point[1]):
                print("Invalid point: {}".format(point[1]))
                continue
            sawyer_robot.set_joint_state(point[1])
            time.sleep(point[0] - prior_time)
            prior_time = point[0]
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    main()
