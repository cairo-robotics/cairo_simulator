from collections import OrderedDict
import json
import pprint
import os
import sys
from functools import partial
import time

if os.environ.get('ROS_DISTRO'):
    import rospy
import numpy as np
import networkx as nx

from cairo_simulator.core.sim_context import SawyerSimContext
from cairo_simulator.core.sim_context import SawyerBiasedTSRSimContext
from cairo_planning.collisions import DisabledCollisionsContext
from cairo_planning.local.interpolation import parametric_lerp
from cairo_planning.local.curve import JointTrajectoryCurve
from cairo_planning.planners import CBiRRT2
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
    # IMPORT FOLIATION DATA FOR CLASSIFICATION  #
    #############################################
    # We will use this information in the       #
    # getFoliation function calls given a set   #
    # of demonstration data                     #
    #############################################


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
    # of constrained motion plans. The goal is  #
    # to create an IPD-Relaxed planning problem #   
    # Each end/start point in the graph will be #
    # a steering point in the Omega set of the  #
    # solution path.                            #
    #############################################
    
    planning_G = nx.Graph()

    start_id = next(iter(keyframes.keys()))
    end_id = next(reversed(keyframes.keys()))

    end_data = [obsv['robot']['joint_angle'] for obsv in keyframes[next(reversed(keyframes.keys()))]["observations"]]
    start_keyframe_dist = KernelDensityDistribution()
    start_keyframe_dist.fit(end_data)

    # Create a distribution for each intermeditate trajectory set.
    # Build into distribution samplers.
    for keyframe_id, keyframe_data in reversed(keyframes.items()):
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


    final_path = []
    sim_context = SawyerSimContext(None, setup=False)
    sim_context.setup(sim_overrides={"run_parallel": False})
    sim = sim_context.get_sim_instance()
    logger = sim_context.get_logger()
    sawyer_robot = sim_context.get_robot()
    svc = sim_context.get_state_validity()
    interp_fn = partial(parametric_lerp, steps=5)
    # sim_obj = SimObject('test', 'r2d2.urdf', (.6, 0.0, .6), fixed_base=1)


    initial_start_point = planning_G.nodes[list(planning_G.nodes)[0]]['point']
    print(initial_start_point)
    for edge in planning_G.edges():
        start_point = planning_G.nodes[edge[0]]['point']
        end_point = planning_G.nodes[edge[1]]['point']
        state_space = planning_G.get_edge_data(*edge)['planning_space']

        ####################################
        # SIMULATION AND PLANNING CONTEXTS #
        ####################################
         with DisabledCollisionsContext(sim, [], [], disable_visualization=True):
            #######
            # LazyPRM #
            #######
            # Use parametric linear interpolation with 10 steps between points.
            interp = partial(parametric_lerp, steps=10)
            # See params for PRM specific parameters
            cbirrt = CBiRRT2(sawyer_robot, state_space, svc, interp, params={'smooth_path': True, 'smoothing_time': 10, 'q_step': .1, 'e_step': .25, 'iters': 20000})
            logger.info("Planning....")
            plan = cbirrt.plan(tsr, np.array(start), np.array(goal))
            path = cbirrt.get_path(plan)
        if len(path) == 0:
            logger.info("Planning failed....")
            sys.exit(1)
        logger.info("Plan found....")

        # splinging uses numpy so needs to be converted
        path = [np.array(p) for p in path]
        logger.info("Length of path: {}".format(len(path)))
        final_path = final_path + path

    sawyer_robot.move_to_joint_pos(initial_start_point)
    time.sleep(3)
    while sawyer_robot.check_if_at_position(initial_start_point, 0.5) is False:
        time.sleep(0.1)
        sim.step()

    key = input("Press any key to excute plan.")

    if len(path) > 0:
        # Create a MinJerk spline trajectory using JointTrajectoryCurve and execute
        jtc = JointTrajectoryCurve()
        traj = jtc.generate_trajectory(final_path, move_time=10)
        sawyer_robot.execute_trajectory(traj)
        try:
            while True:
                sim.step()
        except KeyboardInterrupt:
            sys.exit(0)
    else:
        logger.err("No path found.")
