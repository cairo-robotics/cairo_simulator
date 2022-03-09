import os
import sys
from functools import partial
import time
import json

import pybullet as p
if os.environ.get('ROS_DISTRO'):
    import rospy
import numpy as np

from cairo_simulator.core.sim_context import SawyerBiasedTSRSimContext
from cairo_simulator.core.utils import ASSETS_PATH
from cairo_planning.geometric.transformation import quat2rpy

from cairo_planning.collisions import DisabledCollisionsContext
from cairo_planning.local.interpolation import parametric_lerp
from cairo_planning.local.curve import JointTrajectoryCurve
from cairo_planning.planners import CBiRRT2
from cairo_planning.geometric.state_space import SawyerConfigurationSpace

from cairo_planning.geometric.state_space import DistributionSpace
from cairo_planning.sampling.samplers import DistributionSampler
from cairo_planning.geometric.distribution import KernelDensityDistribution


def main():


    config = {}
    config["sim"] = {
            "use_real_time": False
        }

    config["logging"] = {
            "handlers": ['logging'],
            "level": "debug"
        }

    config["sawyer"] = {
            "robot_name": "sawyer0",
            'urdf_file': ASSETS_PATH + 'sawyer_description/urdf/sawyer_static_mug_combined.urdf',
            "position": [0, 0, 0.9],
            "fixed_base": True
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
            "position": [0.6, 0, .1],
            "orientation":  [0, 0, 1.5708],
            "fixed_base": 1
        },
    ]
    config["primitives"] = [
        {
            "type": "cylinder",
            "primitive_configs": {"radius": .1, "height": .05},
            "sim_object_configs": 
                {
                    "object_name": "cylinder",
                    "position": [.8, -.5726, .6],
                    "orientation":  [0, 0, 0],
                    "fixed_base": 1    
                }
        }
    ]
    config["state_validity"] = {
        "self_collision_exclusions": [("mug", "right_gripper_l_finger"), ("mug", "right_gripper_r_finger")]
    }
    config['tsr'] = {
        'degrees': False,
        "T0_w":  [.7968, -.5772, .15, np.pi/2, -1.3,  np.pi/2],
        "Tw_e":[0, 0, 0, 0, 0, 0],
        "Bw": [[(-.1, .1), (-100, 100), (-100, 100)],  
                [(-100, 100), (-100, 100), (-100, 100)]]
    }
    
    # # Collect all joint configurations from all demonstration .json files.
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
    #     'bandwidth': .1,
    #     'fraction_uniform': .25,
    #     'data': configurations
    # }

    # model = KernelDensityDistribution(bandwidth=config['sampling_bias']['bandwidth'])
    # model.fit(config['sampling_bias']['data'])
    # sampler = DistributionSampler(distribution_model=model, fraction_uniform=config['sampling_bias']['fraction_uniform'])
    # state_space = DistributionSpace(sampler=sampler)

    start = [-0.7048218339245818, 0.45685347725535186, -1.4017091517942437, -0.08979731088870624, -0.06855931921025427, -1.0721452733057917, -0.18843590867040572] 
    goal = [-1.2430838722942696, 0.6191928880135347, -1.2300758302306845, -0.7560430599768746, -0.5663726314028903, -0.6795320204199666, 2.866698285512072]

    sim_context = SawyerBiasedTSRSimContext(configuration=config)
    sim = sim_context.get_sim_instance()
    logger = sim_context.get_logger()
    _ = sim_context.get_state_space()
    sawyer_robot = sim_context.get_robot()
    # _ = sawyer_robot.get_simulator_id()
    tsr = sim_context.get_tsr()
    _ = sim_context.get_sim_objects(['Ground'])[0]
    svc = sim_context.get_state_validity()

    state_space = SawyerConfigurationSpace()
    
    sawyer_robot.set_joint_state(start)
    print(quat2rpy(sawyer_robot.solve_forward_kinematics(start)[0][1]))
    time.sleep(5)
    control = 'g'

    while control != 'q':
        

        with DisabledCollisionsContext(sim, [], [], disable_visualization=True):
            #######
            # LazyPRM #
            #######
            # Use parametric linear interpolation with 10 steps between points.
            interp = partial(parametric_lerp, steps=10)
            # See params for PRM specific parameters
            cbirrt = CBiRRT2(sawyer_robot, state_space, svc, interp, params={'smooth_path': True, 'smoothing_time': 5, 'q_step': .35, 'e_step': .25, 'iters': 100000, 'epsilon': .06})
            logger.info("Planning....")
            plan = cbirrt.plan(tsr, np.array(start), np.array(goal))
            path = cbirrt.get_path(plan)
            print(plan)

            if len(path) == 0:
                logger.info("Planning failed....")
                sys.exit(1)
            logger.info("Plan found....")
            input("Press any key to continue...")
            # splining uses numpy so needs to be converted
            path = [np.array(p) for p in path]
            # Create a MinJerk spline trajectory using JointTrajectoryCurve and execute
            jtc = JointTrajectoryCurve()
            traj = jtc.generate_trajectory(path, move_time=10)
            for i, point in enumerate(traj):
                if not svc.validate(point[1]):
                    print("Invalid point: {}".format(point[1]))
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
        control = input("Press q to quit...")

       


if __name__ == "__main__":
    main()
