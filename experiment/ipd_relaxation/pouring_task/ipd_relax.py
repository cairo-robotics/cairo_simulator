from collections import OrderedDict
import json
import pprint
import os
import sys
from functools import partial
import time
import copy

if os.environ.get('ROS_DISTRO'):
    import rospy
import numpy as np
import networkx as nx

from cairo_simulator.core.utils import ASSETS_PATH
from cairo_simulator.core.sim_context import SawyerBiasedTSRSimContext
from cairo_planning.collisions import DisabledCollisionsContext
from cairo_planning.constraints.foliation import VGMMFoliationClustering, winner_takes_all
from cairo_planning.local.interpolation import parametric_lerp
from cairo_planning.local.curve import JointTrajectoryCurve
from cairo_planning.planners import CBiRRT2
from cairo_planning.geometric.state_space import DistributionSpace
from cairo_planning.geometric.distribution import KernelDensityDistribution
from cairo_planning.sampling.samplers import DistributionSampler

from cairo_planning.geometric.transformation import quat2rpy
from cairo_planning.constraints.projection import project_config
from cairo_planning.geometric.utils import wrap_to_interval


if __name__ == "__main__":
    ######################################
    #       PLANNING CONFIGURATION       #
    ######################################
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
            "orientation":  [0, 0, 1.5708]
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
    # Collect all joint configurations from all demonstration .json files.
    configurations = []
    data_directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), "foliation_data")
    for json_file in os.listdir(data_directory):
        filename = os.path.join(data_directory, json_file)
        with open(filename, "r") as f:
            data = json.load(f)
            for entry in data:
                configurations.append(entry['robot']['joint_angle'])
    
    orientation_foliation_model = VGMMFoliationClustering(estimated_foliations=5)
    orientation_foliation_model.fit(np.array(configurations ))


    ####################################
    # CONSTRAINT TO FOLIATION MAPPING  #
    ####################################

    c2f_map = {}
    c2f_map[(1)] = orientation_foliation_model
    c2f_map[(1, 2)] = orientation_foliation_model

    ##############################
    # CONSTRAINT TO TSR MAPPING  #
    ##############################

    # Let's first define all TSR configurations for this task:

    TSR1_config = {
        'degrees': False,
        "T0_w": [.7, 0, 0, 0, 0, 0],
        "Tw_e": [-.2, 0, .739, np.pi/2, 0,  np.pi/2],
        "Bw": [[(-100, 100), (-100, 100), (-100, 100)],  
                [(-.05, .05), (-.05, .05), (-.05, .05)]]
    }
    # Orientation AND centering constraint
    TSR2_config = {
        'degrees': False,
        "T0_w": [.7, 0, 0, 0, 0, 0],
        "Tw_e": [.7968, -.5726, .739, np.pi/2, 0,  np.pi/2],
        "Bw": [[(-.1, .1), (-.1, .1), (-100, 100)],  
                [(-.05, .05), (-.05, .05), (-.05, .05)]]
    }
    TSR3_config = {
        'degrees': False,
        "T0_w": [.7, 0, 0, 0, 0, 0],
        "Tw_e": [.7968, -.5726, .739, 3.09499115, 0.01804426, 1.59540829],
        "Bw": [[(-.1, .1), (-.1, .1), (-100, 100)],  
                [(-6.3, 6.3), (-6.3, 6.3), (-6.3, 6.3)]]
    }


    c2tsr_map = {}
    c2tsr_map[(1)] = TSR1_config
    c2tsr_map[(1, 2)] = TSR2_config
    c2tsr_map[(2)] = TSR3_config

    #############################################
    #         Import Serialized LfD Graph       #
    #############################################
    with open(os.path.dirname(os.path.abspath(__file__)) + "/lfd_data/lfd_model.json", "r") as f:
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

            # TODO: The keyframe_id and prior_id and choice of inter_traj assignment might need adjustment if we backtrack.

            # Copy the main planning config. This will be updated with specfic configurations for this planning segment (tsrs, biasing etc,.)
            planning_config = copy.deepcopy(config)

            # Create keyframe distrubtion
            data = [obsv['robot']['joint_angle'] for obsv in keyframe_data["observations"]]
            keyframe_dist = KernelDensityDistribution()
            keyframe_dist.fit(data)

            # Let's create the node and add teh keyframe KDE model.
            planning_G.add_nodes_from([(keyframe_id, {"model": keyframe_dist})])



            # get the constraint IDs
            constraint_ids = keyframe_data["applied_constraints"]
            planning_G[keyframe_id]["constraint_ids"] = constraint_ids

            # get the foliation model
            foliation_model = c2f_map.get(tuple(constraint_ids), None)
            planning_G[keyframe_id]["foliation_model"] = foliation_model

            # Determine the foliation choice based on the keyframe data and assign in to the Pg node
            # TODO: Confirm that this value is the same as the upcoming foliation cluster/classification ID.
            # TODO: If not the same, give a warning that the IPD-Relax cannot be gauranteed but proceed anyways?
            foliation_value = winner_takes_all(data, foliation_model)
            planning_G[keyframe_id]["foliation_value"] = foliation_value

            # Get the TSR configurations so they can be appended to the  associated with constraint ID combo.
            planning_config['tsr'] = c2tsr_map.get(tuple(constraint_ids), {})


            # Create intermediate trajectory distribution.
            inter_trajs = intermediate_trajectories[keyframe_id]
            inter_trajs_data = [[obsv['robot']['joint_angle'] for obsv in traj] for traj in inter_trajs]
            
            sampling_bias = {
                'bandwidth': .1,
                'fraction_uniform': .25,
                'data': inter_trajs_data
            }
            planning_config['sampling_bias'] = sampling_bias

            # Finally add the planning config to the planning graph edge. 
            planning_G[keyframe_id][prior_id]['config'] = planning_config


            prior_id = keyframe_id
        if keyframe_id == start_id:
            data = [obsv['robot']['joint_angle'] for obsv in keyframe_data["observations"]]
            keyframe_dist = KernelDensityDistribution()
            keyframe_dist.fit(data)
            planning_G.add_nodes_from([(keyframe_id, {"model": keyframe_dist, "point": data[0]})])
            prior_id = keyframe_id


    final_path = []
    
    # sim_obj = SimObject('test', 'r2d2.urdf', (.6, 0.0, .6), fixed_base=1)


    
    for edge in planning_G.edges():
        config = edge['config']
        sim_context = SawyerBiasedTSRSimContext(config, setup=False)
        sim_context.setup(sim_overrides={"run_parallel": False})
        sim = sim_context.get_sim_instance()
        logger = sim_context.get_logger()
        sawyer_robot = sim_context.get_robot()
        svc = sim_context.get_state_validity()
        tsr = sim_context.get_tsr()
        interp_fn = partial(parametric_lerp, steps=5)
        
        # TODO: Given prior foliations and current foliation, lets generate a steering point using the projection/rejection step.
        start_point = planning_G.nodes[edge[0]]['point']
        
        state_space = planning_G.get_edge_data(*edge)['planning_space']
        n_samples = 5
        valid_samples = []
        with DisabledCollisionsContext(sim, [], []):
            while len(valid_samples) < n_samples:
                sample = state_space.sample()
                if sample is not None and svc.validate(sample):
                    q_constrained = project_config(sawyer_robot, tsr, np.array(
                    sample), np.array(sample), epsilon=.1, e_step=.25)
                    normalized_q_constrained = []
                    print(model.predict(np.array([q_constrained]))[0])
                    print(model.predict(np.array([q_constrained]))[0] == 1)
                    if q_constrained is not None and model.predict(np.array([q_constrained]))[0] == 1:
                        for value in q_constrained:
                            normalized_q_constrained.append(
                                wrap_to_interval(value))
                    else:
                        continue
                    if svc.validate(normalized_q_constrained):
                        valid_samples.append(normalized_q_constrained
        end_point = planning_G.nodes[edge[1]]['point']

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

