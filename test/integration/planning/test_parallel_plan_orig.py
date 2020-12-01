import os
import sys
from functools import partial
import itertools
import time
from multiprocessing import Pool
from timeit import default_timer as timer

import pybullet as p
if os.environ.get('ROS_DISTRO'):
    import rospy
import numpy as np
import igraph as ig

from cairo_simulator.core.simulator import Simulator, SimObject
from cairo_simulator.devices.manipulators import Sawyer
from cairo_simulator.core.log import Logger
from cairo_simulator.core.utils import ASSETS_PATH
from cairo_simulator.core.link import get_link_pairs, get_joint_info_by_name

from cairo_planning.collisions import self_collision_test, DisabledCollisionsContext
from cairo_planning.geometric.state_space import SawyerConfigurationSpace
from cairo_planning.sampling import StateValidityChecker
from cairo_planning.local.evaluation import subdivision_evaluate
from cairo_planning.local.interpolation import parametric_lerp, cumulative_distance
from cairo_planning.local.neighbors import NearestNeighbors
from cairo_planning.constraints.pose_contraints import orientation
from cairo_planning.local.curve import JointTrajectoryCurve


def parallel_sample_worker(num_samples):
    ########################################################
    # Create the Simulator and pass it a Logger (optional) #
    ########################################################
    logger = Logger()
    if not Simulator.is_instantiated():
        logger.warn("Simulator not instantiated, doing so now...")
        sim = Simulator(logger=logger, use_ros=False, use_gui=False,
                        use_real_time=True)  # Initialize the Simulator
    else:
        sim = Simulator.get_instance()

    #####################################
    # Create a Robot, or two, or three. #
    #####################################
    sawyer_robot = Sawyer("sawyer0", [0, 0, 0.9], fixed_base=1)

    #############################################
    # Create sim environment objects and assets #
    #############################################
    ground_plane = SimObject("Ground", "plane.urdf", [0, 0, 0])
    sawyer_id = sawyer_robot.get_simulator_id()

    # Exclude the ground plane and the pedestal feet from disabled collisions.
    excluded_bodies = [ground_plane.get_simulator_id()]  # the ground plane
    pedestal_feet_idx = get_joint_info_by_name(sawyer_id, 'pedestal_feet').idx
    # The (sawyer_idx, pedestal_feet_idx) tuple the ecluded from disabled collisions.
    excluded_body_link_pairs = [(sawyer_id, pedestal_feet_idx)]

    ############
    # SAMPLING #
    ############
    valid_samples = []
    # Disabled collisions during planning with certain eclusions in place.
    with DisabledCollisionsContext(sim, excluded_bodies, excluded_body_link_pairs):
        #########################
        # STATE SPACE SELECTION #
        #########################
        # This inherently uses UniformSampler but a different sampling class could be injected.
        state_space = SawyerConfigurationSpace()

        ##############################
        # STATE VALIDITY FORMULATION #
        ##############################
        # Certain links in Sawyer seem to be permentently in self collision. This is how to remove them by name when getting all link pairs to check for self collision.
        excluded_pairs = [(get_joint_info_by_name(sawyer_id, "right_l1_2").idx, get_joint_info_by_name(sawyer_id, "right_l0").idx),
                          (get_joint_info_by_name(sawyer_id, "right_l1_2").idx, get_joint_info_by_name(sawyer_id, "head").idx)]
        link_pairs = get_link_pairs(sawyer_id, excluded_pairs=excluded_pairs)
        self_collision_fn = partial(
            self_collision_test, robot=sawyer_robot, link_pairs=link_pairs)

        # Create constraint checks
        # def constraint_test(q):
        #     upright_orientation = [
        #         0.0005812598018143569, 0.017721236427960724, -0.6896867930096543, 0.723890701324838]
        #     axis = 'z'
        #     threshold = 15
        #     world_pose, _ = sawyer_robot.solve_forward_kinematics(q)
        #     return orientation(upright_orientation, world_pose[1], threshold, axis)

        # In this case, we only have a self_col_fn.
        svc = StateValidityChecker(
            self_col_func=self_collision_fn, col_func=None, validity_funcs=None)
        # svc = StateValidityChecker(
        #     self_col_func=self_collision_fn, col_func=None, validity_funcs=[constraint_test])

        count = 0
        while count < num_samples:
            q_rand = np.array(state_space.sample())
            if svc.validate(q_rand):
                valid_samples.append(q_rand)
                count += 1
    return valid_samples


def parallel_connect_worker(batches):

    ########################################################
    # Create the Simulator and pass it a Logger (optional) #
    ########################################################
    logger = Logger()
    if not Simulator.is_instantiated():
        sim = Simulator(logger=logger, use_ros=False, use_gui=False,
                        use_real_time=True)  # Initialize the Simulator
    else:
        sim = Simulator.get_instance()

    #####################################
    # Create a Robot, or two, or three. #
    #####################################
    sawyer_robot = Sawyer("sawyer0", [0, 0, 0.9], fixed_base=1)

    #############################################
    # Create sim environment objects and assets #
    #############################################
    ground_plane = SimObject("Ground", "plane.urdf", [0, 0, 0])
    sawyer_id = sawyer_robot.get_simulator_id()

    # Exclude the ground plane and the pedestal feet from disabled collisions.
    excluded_bodies = [ground_plane.get_simulator_id()]  # the ground plane
    pedestal_feet_idx = get_joint_info_by_name(sawyer_id, 'pedestal_feet').idx
    # The (sawyer_idx, pedestal_feet_idx) tuple the ecluded from disabled collisions.
    excluded_body_link_pairs = [(sawyer_id, pedestal_feet_idx)]

    ##########
    # EXTEND #
    ##########
    valid_samples = []
    # Disabled collisions during planning with certain eclusions in place.
    with DisabledCollisionsContext(sim, excluded_bodies, excluded_body_link_pairs):
        #########################
        # STATE SPACE SELECTION #
        #########################
        # This inherently uses UniformSampler but a different sampling class could be injected.
        state_space = SawyerConfigurationSpace()

        ##############################
        # STATE VALIDITY FORMULATION #
        ##############################
        # Certain links in Sawyer seem to be permentently in self collision. This is how to remove them by name when getting all link pairs to check for self collision.
        excluded_pairs = [(get_joint_info_by_name(sawyer_id, "right_l1_2").idx, get_joint_info_by_name(sawyer_id, "right_l0").idx),
                          (get_joint_info_by_name(sawyer_id, "right_l1_2").idx, get_joint_info_by_name(sawyer_id, "head").idx)]
        link_pairs = get_link_pairs(sawyer_id, excluded_pairs=excluded_pairs)
        self_collision_fn = partial(
            self_collision_test, robot=sawyer_robot, link_pairs=link_pairs)

        # Create constraint checks
        # def constraint_test(q):
        #     upright_orientation = [
        #         0.0005812598018143569, 0.017721236427960724, -0.6896867930096543, 0.723890701324838]
        #     axis = 'z'
        #     threshold = 15
        #     world_pose, _ = sawyer_robot.solve_forward_kinematics(q)
        #     return orientation(upright_orientation, world_pose[1], threshold, axis)

        # In this case, we only have a self_col_fn.
        svc = StateValidityChecker(
            self_col_func=self_collision_fn, col_func=None, validity_funcs=None)
        # svc = StateValidityChecker(
        #     self_col_func=self_collision_fn, col_func=None, validity_funcs=[constraint_test])

        pairs = []
        for batch in batches:
            q_sample = batch[0]
            neighbors = batch[1]
            for q_near in neighbors:
                local_path = parametric_lerp(
                    np.array(q_near), np.array(q_sample), steps=10)
                valid = subdivision_evaluate(svc.validate, local_path)
                if valid:
                    pairs.append(
                        [q_near, q_sample, cumulative_distance(local_path)])
        return pairs


if __name__ == "__main__":
    num_workers = 8
    with Pool(num_workers) as p:
        single_s = timer()
        results = p.map(parallel_sample_worker, [10])
        single_e = timer()

    parallel_s = timer()
    with Pool(num_workers) as p:
        multi_s = timer()
        results = p.map(parallel_sample_worker, [
            1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000])
        samples = list(itertools.chain.from_iterable(results))
        multi_e = timer()

    nn = NearestNeighbors(X=np.array(samples), model_kwargs={"leaf_size": 50})
    with Pool(num_workers) as p:
        parallel_connect_s = timer()
        tests = []
        for q_sample in samples:
            distances, neighbors = nn.query(q_sample, k=10)
            q_neighbors = [neighbor for distance, neighbor in zip(
                distances, neighbors) if distance <= 1.5 and distance > 0]
            tests.append((q_sample, q_neighbors))
        batches = np.array_split(tests, num_workers)
        results = p.map(parallel_connect_worker, batches)
        parallel_connect_e = timer()
        connections = list(itertools.chain.from_iterable(results))
        print("{} connections out of {} samples".format(
            len(connections), len(samples)))
    parallel_e = timer()
    print("Single process sampling time {}:".format(single_e - single_s))
    print("Multiprocess sampling time {}:".format(multi_e - multi_s))
    print("Multiprocess connection testing time: {}".format(
        parallel_connect_e - parallel_connect_s))
    print("Multiprocess sampling and connection testing time: {}".format(
        parallel_e - parallel_s))

    ################################
    # Environment Checks and Flags #
    ################################
    if os.environ.get('ROS_DISTRO'):
        rospy.init_node("CAIRO_Sawyer_Simulator")
        use_ros = True
    else:
        use_ros = False

    ########################################################
    # Create the Simulator and pass it a Logger (optional) #
    ########################################################
    logger = Logger()
    sim = Simulator(logger=logger, use_ros=use_ros, use_gui=True,
                    use_real_time=True)  # Initialize the Simulator

    #####################################
    # Create a Robot, or two, or three. #
    #####################################
    sawyer_robot = Sawyer("sawyer0", [0, 0, 0.9], fixed_base=1)
    sawyer_id = sawyer_robot.get_simulator_id()

    #############################################
    # Create sim environment objects and assets #
    #############################################
    ground_plane = SimObject("Ground", "plane.urdf", [0, 0, 0])

    # Exclude the ground plane and the pedestal feet from disabled collisions.
    excluded_bodies = [ground_plane.get_simulator_id()]  # the ground plane
    pedestal_feet_idx = get_joint_info_by_name(sawyer_id, 'pedestal_feet').idx
    # The (sawyer_idx, pedestal_feet_idx) tuple the ecluded from disabled collisions.
    excluded_body_link_pairs = [(sawyer_id, pedestal_feet_idx)]

    ############
    # PLANNING #
    ############
    # Disabled collisions during planning with certain eclusions in place.
    with DisabledCollisionsContext(sim, excluded_bodies, excluded_body_link_pairs):
        #########################
        # STATE SPACE SELECTION #
        #########################
        # This inherently uses UniformSampler but a different sampling class could be injected.
        state_space = SawyerConfigurationSpace()

        ##############################
        # STATE VALIDITY FORMULATION #
        ##############################
        # Certain links in Sawyer seem to be permentently in self collision. This is how to remove them by name when getting all link pairs to check for self collision.
        excluded_pairs = [(get_joint_info_by_name(sawyer_id, "right_l1_2").idx, get_joint_info_by_name(sawyer_id, "right_l0").idx),
                          (get_joint_info_by_name(sawyer_id, "right_l1_2").idx, get_joint_info_by_name(sawyer_id, "head").idx)]
        link_pairs = get_link_pairs(sawyer_id, excluded_pairs=excluded_pairs)
        self_collision_fn = partial(
            self_collision_test, robot=sawyer_robot, link_pairs=link_pairs)
        # In this case, we only have a self_col_fn.
        svc = StateValidityChecker(
            self_col_func=self_collision_fn, col_func=None, validity_funcs=None)

        graph = ig.Graph()

        start = [0, 0, 0, 0, 0, 0, 0]
        end = [1.5262755737449423, -0.1698540226273928, 2.7788151824762055,
               2.4546623466066135, 0.7146948867821279, 2.7671787952787184, 2.606128412644311]

        graph.add_vertex("start")
        # Start is always at the 0 index.
        graph.vs[0]["value"] = start
        graph.add_vertex("goal")
        # Goal is always at the 1 index.
        graph.vs[1]["value"] = end

        for sample in samples:
            graph.add_vertex(None, **{'value': list(sample)})

        for pair in connections:
            graph.add_edge(graph.vs['value'].index(list(pair[0])), graph.vs['value'].index(
                list(pair[1])), **{'weight': pair[2]})

        distances, neighbors = nn.query(np.array(start), k=10)
        for q_near in neighbors:
            if graph.vs['value'].index(q_near) != 0:
                local_path = parametric_lerp(
                    np.array(start), np.array(q_near), steps=10)
                valid = subdivision_evaluate(svc.validate, local_path)
                if valid:
                    graph.add_edge(graph.vs['value'].index(start), graph.vs['value'].index(list(q_near)), **{'weight': pair[2]})
                    break
        distances, neighbors = nn.query(np.array(end), k=10)
        for q_near in neighbors:
            if graph.vs['value'].index(q_near) != 0:
                local_path = parametric_lerp(
                    np.array(start), np.array(q_near), steps=10)
                valid = subdivision_evaluate(svc.validate, local_path)
                if valid:
                    graph.add_edge(graph.vs['value'].index(end), graph.vs['value'].index(list(q_near)), **{'weight': pair[2]})
                    break

        graph_path = graph.get_shortest_paths(
            'start', 'goal', weights='weight', mode='ALL')[0]
        points = [graph.vs[idx]['value'] for idx in graph_path]
        pairs = list(zip(points, points[1:]))
        segments = [parametric_lerp(np.array(p[0]), np.array(p[1]), steps=20)
                    for p in pairs]
        segments = [[list(val) for val in seg] for seg in segments]
        path = []
        for seg in segments:
            path = path + seg

    ##########
    # SPLINE #
    ##########
    # splinging uses numpy so needs to be converted
    path = [np.array(p) for p in path]
    print(path)
    # Create a MinJerk spline trajectory using JointTrajectoryCurve and execute
    jtc = JointTrajectoryCurve()
    traj = jtc.generate_trajectory(path, move_time=10)
    sawyer_robot.execute_trajectory(traj)
    try:
        while True:
            sim.step()
    except KeyboardInterrupt:
        p.disconnect()
        sys.exit(0)