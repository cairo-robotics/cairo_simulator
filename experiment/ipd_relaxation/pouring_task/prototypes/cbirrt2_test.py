import os
import sys
from functools import partial
import time
import json

import pybullet as p

from cairo_planning.constraints.projection import distance_from_TSR
from cairo_planning.sampling import state_validity
if os.environ.get('ROS_DISTRO'):
    import rospy
import numpy as np

from cairo_simulator.core.sim_context import SawyerBiasedSimContext
from cairo_simulator.core.utils import ASSETS_PATH
from cairo_planning.geometric.transformation import quat2rpy
from cairo_planning.geometric.tsr import TSR

from cairo_planning.collisions import DisabledCollisionsContext
from cairo_planning.local.interpolation import parametric_lerp
from cairo_planning.local.curve import JointTrajectoryCurve
from cairo_planning.planners import CBiRRT2
from cairo_planning.geometric.state_space import SawyerConfigurationSpace
from cairo_planning.geometric.transformation import xyzrpy2trans, bounds_matrix, pose2trans

from cairo_planning.geometric.state_space import DistributionSpace
from cairo_planning.sampling.samplers import DistributionSampler
from cairo_planning.geometric.distribution import KernelDensityDistribution
from cairo_planning.geometric.transformation import pose2trans

def distance_to_TSR_config(manipulator, q_s, tsr):
    world_pose, _ = manipulator.solve_forward_kinematics(q_s)
    trans, quat = world_pose[0], world_pose[1]
    T0_s = pose2trans(np.hstack([trans + quat]))
    # generates the task space distance and error/displacement vector
    min_distance_new, x_err = distance_from_TSR(T0_s, tsr)
    return min_distance_new, x_err

def main():


    base_config = {}
    base_config["sim"] = {
            "use_real_time": False
        }

    base_config["logging"] = {
            "handlers": ['logging'],
            "level": "debug"
        }

    base_config["sawyer"] = {
            "robot_name": "sawyer0",
            'urdf_file': ASSETS_PATH + 'sawyer_description/urdf/sawyer_static_mug_combined.urdf',
            "position": [0, 0, 0],
            "fixed_base": True
        }

    base_config["sim_objects"] = [
        {
            "object_name": "Ground",
            "model_file_or_sim_id": "plane.urdf",
            "position": [0, 0, -.9]
        },
        {
            "object_name": "Table",
            "model_file_or_sim_id": ASSETS_PATH + 'table.sdf',
            "position": [0.75, 0, -.8],
            "orientation":  [0, 0, 1.5708],
            "fixed_base": 1
        },
    ]
    base_config["primitives"] = [
        {
            "type": "cylinder",
            "primitive_configs": {"radius": .12, "height": .05},
            "sim_object_configs": 
                {
                    "object_name": "cylinder",
                    "position": [.75, -.6, -.3],
                    "orientation":  [0, 0, 0],
                    "fixed_base": 1    
                }
        },
        {
            "type": "box",
            "primitive_configs": {"w": .25, "l": .25, "h": .45},
            "sim_object_configs": 
                {
                    "object_name": "box",
                    "position": [.75, -.34, -.2],
                    "orientation":  [0, 0, 0],
                    "fixed_base": 1    
                }
        }
    ]
    
    # For the mug-based URDF of sawyer, we need to exclude links that are in constant self collision for the SVC
    base_config["state_validity"] = {
        "self_collision_exclusions": [("mug", "right_gripper_l_finger"), ("mug", "right_gripper_r_finger")]
    }
    
    base_config["tsr"] = {
        'degrees': False,
        "T0_w":  [0.62, -0.6324, 0.15, np.pi/2, -np.pi/2, np.pi/2],
        "Tw_e": [0, 0, 0, 0, 0, 0],
        "Bw": [[(-100, 100), (-100, 100), (0, 100)],  
                [(-.1, .1), (-.1, .1), (-.1, .1)]]
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

    start =  [0.560495990739899, 0.420308558448768, -1.2783427522860364, 0.5076192791954268, 1.7018968770060532, -0.5722165943723141, -1.5264520422711643]
    end = [-0.7564221181204824, 0.09167911976956278, -1.4039700616604895, 0.6753979776749839, -0.0610461738495327, -1.528190623124451, -0.07081692818155894]
    
    sim_context = SawyerBiasedSimContext(configuration=base_config)
    sim = sim_context.get_sim_instance()
    logger = sim_context.get_logger()
    state_space = sim_context.get_state_space()
    sawyer_robot = sim_context.get_robot()
    # _ = sawyer_robot.get_simulator_id()
    _ = sim_context.get_sim_objects(['Ground'])[0]
    svc = sim_context.get_state_validity()
    T0_w = xyzrpy2trans(base_config["tsr"]['T0_w'], degrees=base_config["tsr"]['degrees'])
    Tw_e = xyzrpy2trans(base_config["tsr"]['Tw_e'], degrees=base_config["tsr"]['degrees'])
    Bw = bounds_matrix(base_config["tsr"]['Bw'][0], base_config["tsr"]['Bw'][1])
    tsr = TSR(T0_w=T0_w, Tw_e=Tw_e, Bw=Bw)

    print(distance_to_TSR_config(sawyer_robot, start, tsr))
    print(distance_to_TSR_config(sawyer_robot, end, tsr))
    
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
            interp = partial(parametric_lerp, steps=100)
            # See params for PRM specific parameters
            cbirrt = CBiRRT2(sawyer_robot, state_space, svc, interp, params={'smooth_path': True, 'smoothing_time': 5, 'q_step': .15, 'e_step': .1, 'iters': 100000, 'epsilon': .08})
            logger.info("Planning....")
            plan = cbirrt.plan(tsr, np.array(start), np.array(end))
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
