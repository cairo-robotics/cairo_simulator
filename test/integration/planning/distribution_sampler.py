from collections import OrderedDict
import json
import pprint
import os

import networkx as nx

from cairo_planning.geometric.state_space import DistributionSpace
from cairo_planning.geometric.distribution import KernelDensityDistribution
from cairo_planning.sampling.samplers import DistributionSampler

if __name__ == "__main__":
    #############################################
    #       Important Limits for Samplers       #
    #############################################
    limits = [['right_j0', (-3.0503, 3.0503)],
            ['right_j1', (-3.8095, 2.2736)],
            ['right_j2', (-3.0426, 3.0426)],
            ['right_j3', (-3.0439, 3.0439)],
            ['right_j4', (-2.9761, 2.9761)],
            ['right_j5', (-2.9761, 2.9761)],
            ['right_j6', (-4.7124, 4.7124)],
            ['right_gripper_l_finger_joint', (0.0, 0.020833)],
            ['right_gripper_r_finger_joint',
            (-0.020833, 0.0)],
            ['head_pan', (-5.0952, 0.9064)]]
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
            planning_G.add_nodes_from([(keyframe_id, {"model": keyframe_dist, "point": data[0]})])
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
            planning_G.add_nodes_from([(keyframe_id, {"model": keyframe_dist, "point": data[0]})])
            prior_id = keyframe_id


    for edge in planning_G.edges():
        planning_space = planning_G.get_edge_data(*edge)['planning_space']
        for _ in range(0, 10):
            print(planning_space.sample())

    # create a planner 
    print(planning_G.nodes())
    start_node_data = planning_G.nodes['1']['point']
    edge_data = planning_G['1']['11']
    print(start_node_data)


    

    # Motion plan between the points.
    # For now we will try setting up a plan between two points only. 
    # Connect motion plans and spline between to execute path plans.

