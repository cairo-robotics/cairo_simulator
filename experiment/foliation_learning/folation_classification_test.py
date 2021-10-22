import os
import sys
from collections import OrderedDict
from functools import partial
import time
import json

import networkx as nx
from sklearn.mixture import GaussianMixture
import pybullet as p
if os.environ.get('ROS_DISTRO'):
    import rospy
import numpy as np

from cairo_simulator.core.sim_context import SawyerBiasedTSRSimContext
from cairo_simulator.core.utils import ASSETS_PATH

from cairo_planning.collisions import DisabledCollisionsContext
from cairo_planning.local.interpolation import parametric_lerp
from cairo_planning.local.curve import JointTrajectoryCurve
from cairo_planning.planners import CBiRRT2
from cairo_planning.geometric.state_space import DistributionSpace
from cairo_planning.sampling.samplers import DistributionSampler
from cairo_planning.constraints.foliation import VGMMFoliationClustering



def main():


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
    
    sim_context = SawyerBiasedTSRSimContext(configuration=config)
    sim = sim_context.get_sim_instance()
    logger = sim_context.get_logger()
    # _ = sim_context.get_state_space()
    sawyer_robot = sim_context.get_robot()
    # _ = sawyer_robot.get_simulator_id()
    tsr = sim_context.get_tsr()
    _ = sim_context.get_sim_objects(['Ground'])[0]
    svc = sim_context.get_state_validity()

    # Collect all joint configurations from all demonstration .json files.
    test_configurations = []
    samples_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "samples.json")
  
    with open(samples_file, "r") as f:
        data = json.load(f)
        samples = data['samples']
    print(len(samples))
    test_points = []
    test_points_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_points.json")
  
    with open(test_points_file, "r") as f:
        data = json.load(f)
        for entry in data:
            test_points.append(entry['robot']['joint_angle'])

    model = VGMMFoliationClustering(estimated_foliations=3)
    model.fit(np.array(samples))

    for point in test_points:
        sawyer_robot.set_joint_state(point)
        print(model.predict(np.array([point])))
        time.sleep(5)


    ##########################################################################
    # Let's use the above cluster to guide the choice of steering point.
    # 
    # Working backwards from goal point, and backwards through transition keyframes (and intermediate?)
    # We Look at keyframe data and see which cluster the distribution is closest to. 
        # 1) Sample from the transition keyframe. Sample from the two mixand components
        # 2) Choose the mixand with the smallest Hausdorff distance to the transition keyframe sample set
        # 3) With that chose, sample a point from the transition keyframe and either project or use directly
        # if its already constraint compliant). THIS GIVES US OUR OMEGA/CORRECT steering point.
        # We then create a planning graph structure with these steering points and biasing distributions between steering point (if we want)
    # 
    # We will only use a transition point that        #
    #############################################

    for _ in range(0, 10):
        print(model.sample())


     #############################################
    #         Import Serialized LfD Graph       #
    #############################################
    with open(os.path.dirname(os.path.abspath(__file__)) + "/serialization_test.json", "r") as f:
        serialized_data = json.load(f)
    config = serialized_data["config"]
    intermediate_trajectories = serialized_data["intermediate_trajectories"]
    keyframes = OrderedDict(sorted(serialized_data["keyframes"].items(), key=lambda t: int(t[0])))

    #############################################
    #          CREATE A PLANNING GRAPH          #
    #############################################
    # This will be used to generate a sequence  #
    # of constrained motion plans.              #
    #############################################
    
    planning_G = nx.Graph()
    # Try motion planning using samplers.
    # Generate a set of constraint keyframe waypoints with start end end points included.
    # Get first key

    start_id = next(iter(keyframes.keys()))
    end_id = next(reversed(keyframes.keys()))

    end_data = [obsv['robot']['joint_angle'] for obsv in keyframes[next(reversed(keyframes.keys()))]["observations"]]
    start_keyframe_dist = KernelDensityDistribution()
    start_keyframe_dist.fit(end_data)

    # Create a distribution for each intermeditate trajectory set.
    # Build into distribution samplers.
    for keyframe_id, keyframe_data in keyframes.items():
        if keyframe_data["keyframe_type"] == "constraint_transition" or keyframe_id == end_id:
            # Create keyframe distrubtion
            data = [obsv['robot']['joint_angle'] for obsv in keyframe_data["observations"]]
            keyframe_dist = KernelDensityDistribution()
            keyframe_dist.fit(data)
            # Let's use random keyframe observation point for planning.
            planning_G.add_nodes_from([(keyframe_id, {"model": keyframe_dist, "point": data[0], "type": "constraint_transition"})])
            # Create intermediate trajectory distribution.
            inter_trajs = intermediate_trajectories[keyframe_id]
            inter_trajs_data = [[obsv['robot']['joint_angle'] for obsv in traj] for traj in inter_trajs]
            inter_data = [item for sublist in inter_trajs_data for item in sublist]
            inter_dist = KernelDensityDistribution()
            inter_dist.fit(inter_data)
            planning_G.add_edge(prior_id, keyframe_id)
            planning_G[prior_id][keyframe_id].update({"model": inter_dist})
            planning_space = DistributionSpace(sampler=DistributionSampler(inter_dist), limits=limits)
            planning_G[prior_id][keyframe_id].update({"planning_space": planning_space})
            prior_id = keyframe_id
        if keyframe_id == start_id:
            data = [obsv['robot']['joint_angle'] for obsv in keyframe_data["observations"]]
            keyframe_dist = KernelDensityDistribution()
            keyframe_dist.fit(data)
            planning_G.add_nodes_from([(keyframe_id, {"model": keyframe_dist, "point": data[0], "type": "intermediate"})])
            prior_id = keyframe_id

    # config['sampling_bias'] = {
    #     'bandwidth': .1,
    #     'fraction_uniform': .25,
    #     'data': configurations
    # }

    # model = KernelDensityDistribution(bandwidth=config['sampling_bias']['bandwidth'])
    # model.fit(config['sampling_bias']['data'])
    # sampler = DistributionSampler(distribution_model=model, fraction_uniform=config['sampling_bias']['fraction_uniform'])
    # state_space = DistributionSpace(sampler=sampler)

    # start = [
    #     0.673578125,
    #     -0.2995908203125,
    #     -0.21482421875,
    #     1.4868740234375,
    #     0.53829296875,
    #     0.4117080078125,
    #     -1.2169501953125]

    # goal = [
    #     -1.3020732421875,
    #     -0.44705859375,
    #     0.6508818359375,
    #     1.5064189453125,
    #     -0.889978515625,
    #     0.8245869140625,
    #     -1.6826474609375]

    # sim_context = SawyerBiasedCPRMSimContext(configuration=config)
    # sim = sim_context.get_sim_instance()
    # logger = sim_context.get_logger()
    # # _ = sim_context.get_state_space()
    # sawyer_robot = sim_context.get_robot()
    # # _ = sawyer_robot.get_simulator_id()
    # tsr = sim_context.get_tsr()
    # _ = sim_context.get_sim_objects(['Ground'])[0]
    # svc = sim_context.get_state_validity()

    # sawyer_robot.set_joint_state(start)
    # time.sleep(5)

    # control = 'g'

    # while control != 'q':
        

    #     with DisabledCollisionsContext(sim, [], [], disable_visualization=True):
    #         #######
    #         # LazyPRM #
    #         #######
    #         # Use parametric linear interpolation with 5 steps between points.
    #         interp = partial(parametric_lerp, steps=10)
    #         # See params for PRM specific parameters
    #         cbirrt = CBiRRT2(sawyer_robot, state_space, svc, interp, params={'smooth_path': True, 'smoothing_time': 10, 'q_step': .1, 'e_step': .25, 'iters': 20000})
    #         logger.info("Planning....")
    #         plan = cbirrt.plan(tsr, np.array(start), np.array(goal))
    #         path = cbirrt.get_path(plan)
    #         print(plan)

    #         if len(path) == 0:
    #             logger.info("Planning failed....")
    #             sys.exit(1)
    #         logger.info("Plan found....")
    #         input("Press any key to continue...")
    #         # splining uses numpy so needs to be converted
    #         path = [np.array(p) for p in path]
    #         # Create a MinJerk spline trajectory using JointTrajectoryCurve and execute
    #         jtc = JointTrajectoryCurve()
    #         traj = jtc.generate_trajectory(path, move_time=5)
    #         for i, point in enumerate(traj):
    #             if not svc.validate(point[1]):
    #                 print("Invalid point: {}".format(point[1]))
    #     try:
    #         prior_time = 0
    #         for i, point in enumerate(traj):
    #             if not svc.validate(point[1]):
    #                 print("Invalid point: {}".format(point[1]))
    #                 continue
    #             sawyer_robot.set_joint_state(point[1])
    #             time.sleep(point[0] - prior_time)
    #             prior_time = point[0]
    #     except KeyboardInterrupt:
    #         pass
    #     control = input("Press q to quit...")

       


if __name__ == "__main__":
    main()
