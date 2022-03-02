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
from cairo_simulator.core.sim_context import SawyerBiasedSimContext, SawyerBiasedTSRSimContext

from cairo_planning.collisions import DisabledCollisionsContext
from cairo_planning.constraints.foliation import VGMMFoliationClustering, winner_takes_all
from cairo_planning.constraints.projection import project_config
from cairo_planning.geometric.transformation import xyzrpy2trans, bounds_matrix, quat2rpy
from cairo_planning.geometric.utils import wrap_to_interval
from cairo_planning.geometric.tsr import TSR
from cairo_planning.geometric.state_space import DistributionSpace, SawyerConfigurationSpace
from cairo_planning.geometric.distribution import KernelDensityDistribution
from cairo_planning.local.interpolation import parametric_lerp
from cairo_planning.sampling.samplers import HyperballSampler, DistributionSampler
from cairo_planning.core.serialization import dump_model
from cairo_planning.planners import CPRM

if __name__ == "__main__":
    ###########################################
    #       BASE PLANNING CONFIGURATION       #
    ###########################################
    # Provides consistent baseline configs    #
    # for planning contexts.                  #
    ###########################################
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
            "position": [0, 0, 0.9],
            "fixed_base": True
        }

    base_config["sim_objects"] = [
        {
            "object_name": "Ground",
            "model_file_or_sim_id": "plane.urdf",
            "position": [0, 0, 0]
        },
        {
            "object_name": "Table",
            "model_file_or_sim_id": ASSETS_PATH + 'table.sdf',
            "position": [0.75, 0, .1],
            "orientation":  [0, 0, 1.5708],
            "fixed_base": 1
        },
    ]
    base_config["primitives"] = [
        {
            "type": "cylinder",
            "primitive_configs": {"radius": .1, "height": .05},
            "sim_object_configs": 
                {
                    "object_name": "cylinder",
                    "position": [.9, -.57, .6],
                    "orientation":  [0, 0, 0],
                    "fixed_base": 1    
                }
        }
    ]
    
    # For the mug-based URDF of sawyer, we need to exclude links that are in constant self collision for the SVC
    base_config["state_validity"] = {
        "self_collision_exclusions": [("mug", "right_gripper_l_finger"), ("mug", "right_gripper_r_finger")]
    }

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

    # Collect all joint configurations from all demonstration .json files.
    configurations = []
    data_directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), "foliation_data")
    for json_file in os.listdir(data_directory):
        filename = os.path.join(data_directory, json_file)
        with open(filename, "r") as f:
            data = json.load(f)
            for entry in data:
                configurations.append(entry['robot']['joint_angle'])
    
    # We create a Variational GMM for learning foliations. 
    orientation_foliation_model = VGMMFoliationClustering(estimated_foliations=10)
    orientation_foliation_model.fit(np.array(configurations ))


    ####################################
    # CONSTRAINT TO FOLIATION MAPPING  #
    ####################################
    # Esssentially the only foliations of concern are the orientation, which can dictate whether or not 
    # the skill remains centered. 
    c2f_map = {}
    c2f_map[(1,)] = orientation_foliation_model
    c2f_map[(1, 2)] = orientation_foliation_model
    c2f_map[(1, 3)] = orientation_foliation_model
    c2f_map[(1, 2, 3)] = orientation_foliation_model

    ##############################
    # CONSTRAINT TO TSR MAPPING  #
    ##############################
    # Generic, unconstrained TSR:
    unconstrained_TSR = {
        'degrees': False,
        "T0_w": [0, 0, 0, 0, 0, 0],
        "Tw_e": [.7968, -.5772, .15, np.pi/2, 0,  np.pi/2],
        "Bw": [[(-100, 100), (-100, 100), (-100, 100)],  
                [(-6.3, 6.3), (-6.3, 6.3), (-6.3, 6.3)]]
    }
    # Let's first define all TSR configurations for this task:
    # Orientation only (1)
    TSR_1_config = {
        'degrees': False,
        "T0_w": [0, 0, 0, 0, 0, 0],
        "Tw_e": [.7968, -.5772, .15, np.pi/2, 0,  np.pi/2],
        "Bw": [[(-100, 100), (-100, 100), (-100, 100)],  
                [(-.05, .05), (-.05, .05), (-.05, .05)]]
    }
    # centering only (2)
    TSR_2_config = {
        'degrees': False,
        "T0_w": [0, 0, 0, 0, 0, 0],
        "Tw_e": [.7968, -.5772, .15, np.pi/2, 0,  np.pi/2],
        "Bw": [[(-.1, .1), (-.1, .1), (-100, 100)],  
                [(-6.3, 6.3), (-6.3, 6.3), (-6.3, 6.3)]]
    }
    # height only (3)
    TSR_3_config = {
        'degrees': False,
        "T0_w": [0, 0, 0, 0, 0, 0],
        "Tw_e": [.7968, -.5772, .15, np.pi/2, 0,  np.pi/2],
        "Bw": [[(-100, 100), (-100, 100), (0, 100)],  
                [(-6.3, 6.3), (-6.3, 6.3), (-6.3, 6.3)]]
    }
    # Orientation AND centering constraint (1, 2)
    TSR_12_config = {
        'degrees': False,
        "T0_w": [0, 0, 0, 0, 0, 0],
        "Tw_e": [.7968, -.5772, .739, np.pi/2, 0,  np.pi/2],
        "Bw": [[(-.1, .1), (-.1, .1), (-100, 100)],  
                [(-.05, .05), (-.05, .05), (-.05, .05)]]
    }
    # orientation AND height constraint (1, 3)
    TSR_13_config = {
        'degrees': False,
        "T0_w": [0, 0, 0, 0, 0, 0],
        "Tw_e": [.7968, -.5772, .15, np.pi/2, 0,  np.pi/2],
        "Bw": [[(-100, 100), (-100, 100), (0, 100)],  
                [(-.05, .05), (-.05, .05), (-.05, .05)]]
    }
    # height AND centering constraint (2, 3)
    TSR_23_config = {
        'degrees': False,
        "T0_w": [0, 0, 0, 0, 0, 0],
        "Tw_e": [.7968, -.5772, .15, np.pi/2, 0,  np.pi/2],
        "Bw": [[(-.1, .1), (-.1, .1), (0, 100)],  
                [(-6.3, 6.3), (-6.3, 6.3), (-6.3, 6.3)]]
    }
    # orientation, centering, and height AND height constraint (1, 2, 3)
    TSR_123_config = {
        'degrees': False,
        "T0_w": [0, 0, 0, 0, 0, 0],
        "Tw_e": [.7968, -.5772, .15, np.pi/2, 0,  np.pi/2],
        "Bw": [[(-.1, .1), (-.1, .1), (0, 100)],  
                [(-.05, .05), (-.05, .05), (-.05, .05)]]
    }

    c2tsr_map = {}
    c2tsr_map[(1,)] = TSR_1_config
    c2tsr_map[(2,)] = TSR_2_config
    c2tsr_map[(3,)] = TSR_3_config
    c2tsr_map[(1, 2)] = TSR_12_config
    c2tsr_map[(1, 3)] = TSR_13_config
    c2tsr_map[(2, 3)] = TSR_23_config
    c2tsr_map[(1, 2, 3)] = TSR_123_config

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
    
    # The start configuration. For this pouring task, we let the keyframe model / sampling dicate the ending point (but this is not strictly required)
    start_configuration = [0.4523310546875, 0.8259462890625, -1.3458369140625, 0.3512138671875, 1.7002646484375, -0.7999306640625, -1.324783203125]
    planning_G = nx.Graph()

    # Starting and ending keyframe ids
    start_keyframe_id = list(keyframes.keys())[0]
    end_keyframe_id = list(keyframes.keys())[-1]

    ###############################################################################
    # Insert last keyframe into planning graph before looping over keyframe model #
    ###############################################################################
    
    # We will build a keyframe dsitribution using KDE from which to sample for steering points / viapoints. 
    end_data = [obsv['robot']['joint_angle'] for obsv in keyframes[end_keyframe_id]["observations"]]
    keyframe_dist = KernelDensityDistribution(bandwidth=.025)
    keyframe_dist.fit(end_data)
    keyframe_space = DistributionSpace(sampler=DistributionSampler(keyframe_dist, fraction_uniform=0), limits=limits)
    # we cast the keyframe ids to int for networkx node dereferencing as keyframe ids are output as strings from CAIRO LfD 
    planning_G.add_nodes_from([int(end_keyframe_id)], keyframe_space=keyframe_space)

    # get the constraint IDs
    constraint_ids = keyframes[end_keyframe_id]["applied_constraints"]
    foliation_constraint_ids = list(set(keyframes[end_keyframe_id]["applied_constraints"] + keyframes[list(keyframes.keys())[-2]]["applied_constraints"]))
    planning_G.nodes[int(end_keyframe_id)]["constraint_ids"] = constraint_ids
    
    # get the foliation model
    foliation_model = c2f_map.get(tuple(sorted(foliation_constraint_ids)), None)
    
    # there's a possibility that the ending keyframe is not constrained and thus might not provide a foliation model to use
    if foliation_model is not None:
        planning_G.nodes[int(end_keyframe_id)]["foliation_model"] = foliation_model

        # Determine the foliation choice based on the keyframe data and assign in to the planning graph node
        # Winner takes all essentially chooses the most common foliation value base on classifying each data point
        foliation_value = winner_takes_all(data, foliation_model)
        upcoming_foliation_value = foliation_value
    
        planning_G.nodes[int(end_keyframe_id)]["foliation_value"] = foliation_value
    else:
        upcoming_foliation_value = None

    # Get the TSR configurations so they can be appended to the  associated with constraint ID combo.
    planning_G.nodes[int(end_keyframe_id)]['tsr'] = c2tsr_map.get(tuple(sorted(constraint_ids)), unconstrained_TSR)
    
    # the end id will be the first upcoming ID
    upcoming_id = int(end_keyframe_id)
        
        
    ############################################################################
    # Reverse iteration over the keyframe model to populate our planning graph #
    ############################################################################
    
    reversed_keyframes = list(reversed(keyframes.items()))[1:]
    
    # used to keep track of sequence of constraint transition, start, and end keyframe ids as
    # not all keyframes in the lfd model will be used
    keyframe_planning_order = []
    keyframe_planning_order.insert(0, int(end_keyframe_id))
    
    # iteration over the list of keyframes in reverse order ecluding the last keyframe ID which has been handled above
    for idx, item in enumerate(reversed_keyframes):
        keyframe_id = int(item[0])
        keyframe_data = item[1]
        # We only use constraint transition, start, and end keyframes (which has already been accounted for above).
        if keyframe_data["keyframe_type"] == "constraint_transition" or keyframe_id == int(start_keyframe_id):
            
            # We keep track of the sequence of keyframe ids in order to established a static ordering of keyframe pairs to plan between
            keyframe_planning_order.insert(0, keyframe_id)

            # Copy the base planning config. This will be updated with specfic configurations for this planning segment (tsrs, biasing etc,.)
            planning_config = copy.deepcopy(base_config)

            # Create KDE distrubtion for the current keyframe.
            data = [obsv['robot']['joint_angle'] for obsv in keyframe_data["observations"]]
            keyframe_dist = KernelDensityDistribution(bandwidth=.025)
            keyframe_dist.fit(data)
            # We want to fully bias sampling from keyframe distributions.
            keyframe_space = DistributionSpace(sampler=DistributionSampler(keyframe_dist, fraction_uniform=0), limits=limits)

            # Let's create the node and add teh keyframe KDE model as a planning space.
            planning_G.add_nodes_from([keyframe_id], keyframe_space=keyframe_space)

            # get the constraint IDs
            constraint_ids = keyframe_data["applied_constraints"]
            
            # The foliation constraint ids combines both start and end keyframes of the planning segment. In other words, we need to 
            # ensure the start point and ending steering point are in the same foliation, so we utilize the constraints from both keyframes.
            foliation_constraint_ids = list(set(keyframe_data["applied_constraints"] + keyframes[str(upcoming_id)]["applied_constraints"]))

            planning_G.nodes[keyframe_id]["constraint_ids"] = constraint_ids
            
            # Get the TSR configurations so they can be appended to both the keyframe and the edge between associated with constraint ID combo.
            planning_G.nodes[keyframe_id]['tsr'] = c2tsr_map.get(tuple(sorted(constraint_ids)), unconstrained_TSR)
            planning_config['tsr'] = c2tsr_map.get(tuple(sorted(constraint_ids)), unconstrained_TSR)
            
            # get the foliation model
            foliation_model = c2f_map.get(tuple(sorted(foliation_constraint_ids)), None)
            if foliation_model is not None:
                # Assign the foliation model to the planning graph node for the current keyframe.
                planning_G.nodes[keyframe_id]["foliation_model"] = foliation_model
                # Determine the foliation choice based on the keyframe data and assign in to the Pg node
                foliation_value = winner_takes_all(data, foliation_model)
                # We want the current foliation value / component to be equivalent to the upcoming foliation value
                # TODO: Integrate equivalency set information so that based on trajectories, we have some confidence if to 
                # foliation values are actually from the same foliation value. 
                if foliation_value != upcoming_foliation_value and upcoming_foliation_value is not None:
                    print("Foliation values are not equivalent, cannot gaurantee planning feasibility but will proceed.")
                planning_G.nodes[keyframe_id]["foliation_value"] = foliation_value
                upcoming_foliation_value = foliation_value
            # If the current keyframe doesn't have a constraint with an associated model then we dont care to hold 
            # onto the upcoming foliation value any longer
            else:
                upcoming_foliation_value = None

            
            if keyframe_id != int(start_keyframe_id):
                # Create intermediate trajectory ditribution configuration.
                inter_trajs = intermediate_trajectories[str(keyframe_id)]
                inter_trajs_data = []
                for traj in inter_trajs:
                    inter_trajs_data = inter_trajs_data + [obsv['robot']['joint_angle'] for obsv in traj]
                
                # this information will be used to create a biasing distribution for sampling during planning between steering points.
                sampling_bias = {
                    'bandwidth': .1,
                    'fraction_uniform': .1,
                    'data': inter_trajs_data
                }
                planning_config['sampling_bias'] = sampling_bias

            planning_G.add_edge(keyframe_id, upcoming_id)
            # Finally add the planning config to the planning graph edge. 
            planning_G.edges[keyframe_id, upcoming_id]['config'] = planning_config

            # update the upcoming keyframe id with the current id
            upcoming_id = keyframe_id
 

    # Let's insert the starting point:
    # Copy the base planning config. This will be updated with specfic configurations for this planning segment (tsrs, biasing etc,.)
    planning_config = copy.deepcopy(base_config)
    # We populat ethe "point" attribute of the planning graph node which will indicate that we do not need to sample from this node
    # We also use a basic keyframe space -> TODO: is this necessary?
    planning_G.add_nodes_from([(0, {"point": start_configuration, "keyframe_space": SawyerConfigurationSpace(limits=limits)})])
    planning_G.nodes[0]['tsr'] = TSR_1_config
    # let's connect the starting point to the node associated with the starting keyframe
    planning_G.add_edge(0, int(start_keyframe_id))
    keyframe_planning_order.insert(0, 0)
    planning_config['tsr'] = TSR_1_config
    planning_G.nodes[int(start_keyframe_id)]['tsr'] = TSR_1_config
    # Add the lanning config to the planning graph edge. 
    planning_G.edges[0, int(start_keyframe_id)]['config'] = planning_config
    # A list to append path segments in order to create one continuous path
    final_path = []
    
    ###################################################
    #           SEQUENTIAL MANIFOLD PLANNING          #
    ###################################################
    # Now that we've defined our planning problem     #
    # withing a planning graph, which defines our SMP #
    # problem. We perform IPD relaxation and actual   #
    # planning.                                       #
    ###################################################
    
    # Here we use the keyframe planning order, creating a sequential pairing of keyframe ids.
    for edge in list(zip(keyframe_planning_order, keyframe_planning_order[1:])):
        e1 = edge[0]
        e2 = edge[1]
        edge_data = planning_G.edges[e1, e2]
        # lets ge the planning config from the edge or use the generic base config defined above
        config = edge_data.get('config', base_config)
        
        # We create a Sim context from the config for planning. 
        sim_context = SawyerBiasedTSRSimContext(config, setup=False)
        sim_context.setup(sim_overrides={"use_gui": False, "run_parallel": False})
        planning_state_space = sim_context.get_state_space() # The biased state space for sampling points according to intermediate trajectories.
        sim = sim_context.get_sim_instance()
        logger = sim_context.get_logger()
        sawyer_robot = sim_context.get_robot()
        svc = sim_context.get_state_validity() # the SVC is the same for all contexts so we will use this one in our planner.
        interp_fn = partial(parametric_lerp, steps=20)

        # Create the TSR object
        planning_tsr_config =  planning_G.nodes[e1].get("tsr", unconstrained_TSR)
        T0_w = xyzrpy2trans(planning_tsr_config['T0_w'], degrees=planning_tsr_config['degrees'])
        Tw_e = xyzrpy2trans(planning_tsr_config['Tw_e'], degrees=planning_tsr_config['degrees'])
        Bw = bounds_matrix(planning_tsr_config['Bw'][0], planning_tsr_config['Bw'][1])
        # we plan with the current edges first/starting node's tsr and planning space.
        planning_tsr = TSR(T0_w=T0_w, Tw_e=Tw_e, Bw=Bw, bodyandlink=0, manipindex=16)
        keyframe_space_e1 = planning_G.nodes[e1]['keyframe_space']
        
        # generate a starting point, and a steering point, according to constraints (if applicable). 
        # check if the starting point has generated already:
        if  planning_G.nodes[e1].get('point', None) is None:
            print("Constraints {}}: {}".format(e1, planning_G.nodes[e1].get("constraint_ids", [])))
            foliation_model =  planning_G.nodes[e1].get("foliation_model", None)
            foliation_value =  planning_G.nodes[e1].get("foliation_value", None)

            with DisabledCollisionsContext(sim, [], [], disable_visualization=True):
                found = False
                while not found:
                    raw_sample = keyframe_space_e1.sample()
                    sample = []
                    for value in raw_sample:
                        sample.append(wrap_to_interval(value))
                    # If the sample is already constraint compliant, no need to project. Thanks LfD!
                    xyz, rpy = sawyer_robot.solve_forward_kinematics(sample)[0]
                    pose = list(xyz) + list(quat2rpy(quat))
                    if not all(planning_tsr.is_valid(pose)) and svc.validate(sample):
                        if sample is not None and svc.validate(sample):
                            q_constrained = project_config(sawyer_robot, planning_tsr, np.array(
                            sample), np.array(sample), epsilon=.1, e_step=.35, q_step=100)
                            normalized_q_constrained = []
                            # If there is a foliation model, then we must perform rejection sampling until the projected sample is classified 
                            # to the node's foliation value
                            if q_constrained is not None:
                                if foliation_model is not None:
                                    # This is the rejection sampling step to enforce the foliation choice
                                    if foliation_model.predict(np.array([q_constrained])) == foliation_value:
                                        for value in q_constrained:
                                            normalized_q_constrained.append(
                                                wrap_to_interval(value))
                                    else:
                                        continue
                                else:
                                    for value in q_constrained:
                                            normalized_q_constrained.append(
                                                wrap_to_interval(value))
                            else:
                                continue
                            if svc.validate(normalized_q_constrained):
                                start = normalized_q_constrained
                                # We've generated a point so lets use it moving forward for all other planning segments. 
                                planning_G.nodes[e1]['point'] = start
                                found = True
                    else:
                        start = sample
                        planning_G.nodes[e1]['point'] = start
                        found = True
        # if the point steering/start point is available already, then we simply use it 
        else:
            start = planning_G.nodes[e1]['point']
            print("Using stored e1 point")
            print(start)

        if e2 == 33 or e2 == '33':
            print("Made it")

        if  planning_G.nodes[e2].get('point', None) is None:
            keyframe_space_e2 =  planning_G.nodes[e2]['keyframe_space']
            tsr_config =  planning_G.nodes[e2].get("tsr", unconstrained_TSR)
            T0_w2 = xyzrpy2trans(tsr_config['T0_w'], degrees=tsr_config['degrees'])
            Tw_e2 = xyzrpy2trans(tsr_config['Tw_e'], degrees=tsr_config['degrees'])
            Bw2 = bounds_matrix(tsr_config['Bw'][0], tsr_config['Bw'][1])
            tsr = TSR(T0_w=T0_w2, Tw_e=Tw_e2, Bw=Bw2, bodyandlink=0, manipindex=16)
            
            print("Constraints {}: {}".format(e2, planning_G.nodes[e1].get("constraint_ids", [])))
            foliation_model =  planning_G.nodes[e2].get("foliation_model", None)
            foliation_value =  planning_G.nodes[e2].get("foliation_value", None)

            with DisabledCollisionsContext(sim, [], [], disable_visualization=True):
                found = False
                while not found:
                    raw_sample = keyframe_space_e2.sample()
                    sample = []
                    for value in raw_sample:
                        sample.append(wrap_to_interval(value))
                    # If the sample is already constraint compliant, no need to project. Thanks LfD!
                    xyz, quat = sawyer_robot.solve_forward_kinematics(sample)[0]
                    pose = list(xyz) + list(quat2rpy(quat))
                    if not all(planning_tsr.is_valid(pose)) and svc.validate(sample):
                        if sample is not None and svc.validate(sample):
                            # We use a large q_step since we're not using project_config for cbirrt2 but instead just trying to project as ingle point. We don't care about how far we're stepping w.r.t tree growth
                            q_constrained = project_config(sawyer_robot, tsr, np.array(
                            sample), np.array(sample), epsilon=.1, e_step=.25, q_step=100)
                            normalized_q_constrained = []
                            if q_constrained is not None:
                                if foliation_model is not None:
                                    # This is the rejection sampling step to enforce the foliation choice
                                    if foliation_model.predict(np.array([q_constrained])) == foliation_value:
                                        for value in q_constrained:
                                            normalized_q_constrained.append(
                                                wrap_to_interval(value))
                                    else:
                                        continue
                                else:
                                    for value in q_constrained:
                                            normalized_q_constrained.append(
                                                wrap_to_interval(value))
                            else:
                                continue
                            if svc.validate(normalized_q_constrained):
                                end = normalized_q_constrained
                                # We've generated a point so lets use it moving forward for all other planning segments. 
                                planning_G.nodes[e2]['point'] = end
                                found = True
                    else:
                        end = sample
                        planning_G.nodes[e2]['point'] = end
                        found = True
        else:
            end = planning_G.nodes[e2]['point']
        print("DISTANCE: ", np.linalg.norm(np.array(start) - np.array(end)))
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
            print(config)
            time.sleep(10)
            prm = CPRM(SawyerBiasedTSRSimContext, config, sawyer_robot, planning_tsr, planning_state_space, tree_state_space, svc, interp, params={
                'n_samples': 300, 'k': 15, 'planning_attempts': 5, 'ball_radius': 10.0, 'smooth_path': True, 'cbirrt2_sampling_space': 'hyperball', 'smoothing_time': 5}, tree_params={'iters': 200, 'q_step': .48, 'e_step': .25})
            logger.info("Planning....")
            ptime1 = time.process_time()
            path = prm.generate_roadmap(np.array(start), np.array(end))
            # Dump the samples and configuration
            dump_model(sim_context.config, prm, os.path.dirname(os.path.abspath(__file__)), directory_name="{}_{}-".format(int(e1), int(e2)))

        
                
