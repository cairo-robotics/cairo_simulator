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
from cairo_planning.core.planning_context import SawyerPlanningContext
from cairo_planning.collisions import DisabledCollisionsContext
from cairo_planning.local.interpolation import parametric_lerp
from cairo_planning.local.curve import JointTrajectoryCurve
from cairo_planning.planners import PRM
from cairo_planning.geometric.state_space import SawyerTSRConstrainedSpace
from cairo_planning.geometric.distribution import KernelDensityDistribution
from cairo_planning.sampling.samplers import DistributionSampler
from cairo_planning.geometric.transformation import xyzrpy2trans, bounds_matrix, quat2rpy
from cairo_planning.geometric.tsr import TSR
from cairo_planning.geometric.utils import geodesic_distance, wrap_to_interval

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


    sim_context = SawyerSimContext(None, setup=False, planning_context=None)
    sim_context.setup(sim_overrides={"run_parallel": False})
    sim = sim_context.get_sim_instance()
    logger = sim_context.get_logger()
    sawyer_robot = sim_context.get_robot()
    svc = sim_context.get_state_validity()
    interp_fn = partial(parametric_lerp, steps=5)

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
    constraint_transition_count = 0
    for keyframe_id, keyframe_data in keyframes.items():
        print(keyframe_id)
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

            # Utilizes RPY convention
            T0_w = xyzrpy2trans([.7, 0, 0, 0, 0, 0], degrees=False)

            # Utilizes RPY convention
            Tw_e = xyzrpy2trans([-.2, 0, 1.0, np.pi/2, np.pi, 0], degrees=False)

            if constraint_transition_count == 0:
                Bw = bounds_matrix([(-100, 100), (-100, 100), (-100, 100)],  # No positional constraint bounds.
                            [(-np.pi, np.pi), (-np.pi, np.pi), (-np.pi, np.pi)])  # any rotation about z, with limited rotation about x, and y.
                constraint_transition_count += 1
            else:
                print("using constraint bounds")
                # Utilizes RPY convention
                Bw = bounds_matrix([(-100, 100), (-100, 100), (-100, 100)],  # No positional constraint bounds.
                                [(-.07, .07), (-.07, .07), (-.07, .07)])  # any rotation about z, with limited rotation about x, and y.
            tsr = TSR(T0_w=T0_w, Tw_e=Tw_e, Bw=Bw,
                    manipindex=0, bodyandlink=16)

            planning_space = SawyerTSRConstrainedSpace(sampler=DistributionSampler(inter_dist), limits=limits, svc=svc, TSR=tsr, robot=sawyer_robot)
            planning_G[prior_id][keyframe_id].update({"planning_space": planning_space})
            prior_id = keyframe_id
        if keyframe_id == start_id:
            data = [obsv['robot']['joint_angle'] for obsv in keyframe_data["observations"]]
            keyframe_dist = KernelDensityDistribution()
            keyframe_dist.fit(data)
            planning_G.add_nodes_from([(keyframe_id, {"model": keyframe_dist, "point": data[0]})])
            prior_id = keyframe_id


    final_path = []

    initial_start_point = planning_G.nodes[list(planning_G.nodes)[0]]['point']
  
    for edge in planning_G.edges():
        valid_samples = []

        
        start_point = planning_G.nodes[edge[0]]['point']
        end_point = planning_G.nodes[edge[1]]['point']
        print(start_point, end_point)
        state_space = planning_G.get_edge_data(*edge)['planning_space']

        ####################################
        # SIMULATION AND PLANNING CONTEXTS #
        ####################################
        with DisabledCollisionsContext(sim, [], []):
            #######
            # PRM #
            #######
            # Use parametric linear interpolation with 10 steps between points.
            interp = partial(parametric_lerp, steps=10)
            # See params for PRM specific parameters
            prm = PRM(state_space, svc, interp_fn, params={
                    'n_samples': 250, 'k': 6, 'ball_radius': .75})
            logger.info("Planning....")
            plan = prm.plan(np.array(start_point), np.array(end_point))
            # get_path() reuses the interp function to get the path between vertices of a successful plan
            path = prm.get_path(plan)
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
    time.sleep(3)

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

