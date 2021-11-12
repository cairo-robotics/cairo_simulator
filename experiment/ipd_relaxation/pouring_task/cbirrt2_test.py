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
    config['tsr'] = {
        'degrees': False,
        "T0_w": [0, 0, 0, 0, 0, 0],
        "Tw_e": [.6437, -.5772, .15, np.pi/2, 0,  np.pi/2],
        "Bw": [[(-.1, .1), (-.1, .1), (-100, 100)],  
                [(-6.3, 6.3), (-6.3, 6.3), (-6.3, 6.3)]]
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

    start = [-0.8901298400178792, 0.4028941410562896, -1.3139975650868203, 0.26779303203209137, -0.14711643851447542, -1.4349792581936127, -0.27535531869973573] 
    goal = [-1.3204637331062832, 0.7306786975256818, -1.2184963669194724, -0.7138895160443961, -0.5016879537781676, -0.6318574521723868, 2.8874924188413242]

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
            # Use parametric linear interpolation with 5 steps between points.
            interp = partial(parametric_lerp, steps=50)
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
