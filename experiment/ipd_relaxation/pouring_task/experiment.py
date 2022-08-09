from collections import OrderedDict
import json
import argparse
import os
import sys
from functools import partial
import time
import copy
import logging
from pathlib import Path

if os.environ.get('ROS_DISTRO'):
    import rospy
import numpy as np
import networkx as nx

from cairo_simulator.core.utils import ASSETS_PATH
from cairo_simulator.core.sim_context import SawyerBiasedSimContext

from cairo_planning.collisions import DisabledCollisionsContext
from cairo_planning.constraints.foliation import VGMMFoliationClustering, winner_takes_all
from cairo_planning.core.serialization import load_json_files
from cairo_planning.evaluation.eval import IPDRelaxEvaluationTrial, IPDRelaxEvaluation
from cairo_planning.geometric.transformation import xyzrpy2trans, bounds_matrix, pose2trans
from cairo_planning.geometric.utils import wrap_to_interval
from cairo_planning.geometric.tsr import TSR
from cairo_planning.geometric.state_space import DistributionSpace, SawyerConfigurationSpace
from cairo_planning.geometric.distribution import KernelDensityDistribution
from cairo_planning.local.interpolation import parametric_lerp
from cairo_planning.local.curve import JointTrajectoryCurve
from cairo_planning.planners import CBiRRT2
from cairo_planning.planners.exceptions import PlanningTimeoutException, MaxItersException
from cairo_planning.sampling.samplers import DistributionSampler

from cairo_planning.constraints.projection import distance_from_TSR

from cairo_planning_core import Agent

FILE_DIR = os.path.dirname(os.path.realpath(__file__))

def distance_to_TSR_config(manipulator, q_s, tsr):
    world_pose, _ = manipulator.solve_forward_kinematics(q_s)
    trans, quat = world_pose[0], world_pose[1]
    T0_s = pose2trans(np.hstack([trans + quat]))
    # generates the task space distance and error/displacement vector
    min_distance_new, x_err = distance_from_TSR(T0_s, tsr)
    return min_distance_new, x_err

script_logger = logging.getLogger("omega_optimize")
script_logger.setLevel(logging.INFO)
fh = logging.FileHandler('omega_optimize.log', 'a')
fh.setLevel(logging.INFO)
# create formatter and add it to the handlers
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
# add the handlers to logger
script_logger.addHandler(fh)

if __name__ == "__main__":
    
    ########
    # ARGS #
    ########
    arg_fmt = argparse.ArgumentDefaultsHelpFormatter
    parser = argparse.ArgumentParser(formatter_class=arg_fmt)
    parser.add_argument(
        "-p", "--participant", dest="participant", default="1",
        choices=["1", "2", "3"],
        help='Which of the three participants to run: "1", "2" or "3"'
        )
    parser.add_argument(
        "-b", "--biased_planning", dest="biased_planning", default="1",
        choices=["0", "1"],
        help='Use demonstration data as a bias (1) or not (0).'
        )
    parser.add_argument(
        "-i", "--ip_style", dest="ip_style", default="optkf",
        choices=["optkf", "opt", "kf"],
        help='Designate which method of steering point generation: Optimization + keyframe ("optkf"), optimization-only ("opt"), keyframe-only ("kf").'
        )
    parser.add_argument(
        "-c", "--use_collision_objects", dest="use_collision_objects", default="0",
        choices=["0", "1"],
        help='Indicate whether to use '
        )
    args = parser.parse_args()
    participant = args.participant # determines input/output paths
    biased_planning = 1 if args.biased_planning == "1" else 0
    ip_style = args.ip_style
    use_collision_objects = 1 if args.use_collision_objects == "1" else 0


    ######################################
    # CONSTANTS / VARIABLES / PARAMETERS #
    ######################################
    
    # DIRECTORIES
    FOLIATION_DATA_DIRECTORY = os.path.join(FILE_DIR, "participant_{}/foliation_data".format(participant))
    LFD_MODEL_FILEPATH = os.path.join(FILE_DIR, "participant_{}/lfd_data/lfd_model.json".format(participant))
    GOLD_DEMO_INPUT_DIRECTORY = os.path.join(FILE_DIR, "participant_{}/gold_demo/*.json".format(participant))
    EVAL_OUTPUT_DIRECTORY = os.path.join(FILE_DIR, "participant_{}/output".format(participant))

    # IPD RELAX PARAMS
    KEYFRAME_KDE_BANDWIDTH = .35
    SAMPLING_BIAS_KDE_BANDWIDTH = .05
    OPTIMIZATION_ITERS = 1000
    OMEGA_TSR_EPSILON = .075
    
    # PLANNING PARAMS
    SMOOTH = False
    SMOOTHING_TIME = 10
    MAX_SEGMENT_PLANNING_TIME = 60
    MAX_ITERS = 5000
    PLANNING_TSR_EPSILON = .1
    Q_STEP = .05
    E_STEP = .025
    MOVE_TIME = 10
    
    # EXPERIMENT PARAMS
    TRIALS = 10
    VISUALIZE_EXECUTION = False
       
    ##############
    # EVALUATION #
    ##############
    
    # The evaluation object.
    evaluation = IPDRelaxEvaluation(EVAL_OUTPUT_DIRECTORY, participant=participant, biased_planning=biased_planning, ip_style=ip_style)
        
    # Create the gold demonstration trajectory
    gold_demo_data = load_json_files(GOLD_DEMO_INPUT_DIRECTORY)["data"][0]
    gold_demo_traj = [[entry["robot"]["joint_angle"] for entry in gold_demo_data]]
    
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
    
    
    # ADditional primitives to serve as collision object conditions
    
    conditional_collision_objects = [
        {
            "type": "cylinder",
            "primitive_configs": {"radius": .12, "height": .15},
            "sim_object_configs": 
                {
                    "object_name": "cylinder",
                    "position": [.2, -.6, -.3],
                    "orientation":  [0, 0, 0],
                    "fixed_base": 1    
                }
        }
    ]
    
    if use_collision_objects:
        base_config["primitives"] = base_config["primitives"] + conditional_collision_objects
    
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
    script_logger.info("Creating foliation VGMM Model")
    # Collect all joint configurations from all demonstration .json files.
    configurations = []
    for json_file in os.listdir(FOLIATION_DATA_DIRECTORY):
        filename = os.path.join(FOLIATION_DATA_DIRECTORY, json_file)
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
        "T0_w":  [0.62, -0.6324,  0.15, np.pi/2, -np.pi/2, np.pi/2],
        "Tw_e": [0, 0, 0, 0, 0, 0],
        "Bw": [[(-100, 100), (-100, 100), (-100, 100)],  
                [(-100, 100), (-100, 100), (-100, 100)]]
    }
    # Let's first define all TSR configurations for this task:
    # Orientation only (1)
    TSR_1_config = {
        'degrees': False,
        "T0_w":  [0.62, -0.6324, 0.15, np.pi/2, -np.pi/2, np.pi/2],
        "Tw_e": [0, 0, 0, 0, 0, 0],
        "Bw": [[(-100, 100), (-100, 100), (-100, 100)],  
                [(-.1, .1), (-.1, .1), (-.1, .1)]]
    }
    # centering only (2)
    TSR_2_config = {
        'degrees': False,
        "T0_w":  [0.62, -0.6324, 0.15, np.pi/2, -np.pi/2, np.pi/2],
        "Tw_e": [0, 0, 0, 0, 0, 0],
        "Bw": [[(-.05, .05), (-.05, .05), (-100, 100)],  
                [(-100, 100), (-100, 100), (-100, 100)]]
    }
    # height only (3)
    TSR_3_config = {
        'degrees': False,
        "T0_w":  [0.62, -0.6324, 0.15, np.pi/2, -np.pi/2, np.pi/2],
        "Tw_e": [0, 0, 0, 0, 0, 0],
        "Bw": [[(-100, 100), (-100, 100), (0, 100)],  
                 [(-100, 100), (-100, 100), (-100, 100)]]
    }
    # Orientation AND centering constraint (1, 2)
    TSR_12_config = {
        'degrees': False,
        "T0_w":  [0.62, -0.6324, 0.15, np.pi/2, -np.pi/2, np.pi/2],
        "Tw_e": [0, 0, 0, 0, 0, 0],
        "Bw": [[(-.05, .05), (-.05, .05), (-100, 100)],  
                [(-.1, .1), (-.1, .1), (-.1, .1)]]
    }
    # orientation AND height constraint (1, 3)
    TSR_13_config = {
        'degrees': False,
        "T0_w":  [0.62, -0.6324, 0.15, np.pi/2, -np.pi/2, np.pi/2],
        "Tw_e": [0, 0, 0, 0, 0, 0],
        "Bw": [[(-100, 100), (-100, 100), (0, 100)],  
                [(-.1, .1), (-.1, .1), (-.1, .1)]]
    }
    # height AND centering constraint (2, 3)
    TSR_23_config = {
        'degrees': False,
        "T0_w":  [0.62, -0.6324, 0.15, np.pi/2, -np.pi/2, np.pi/2],
        "Tw_e": [0, 0, 0, 0, 0, 0],
        "Bw": [[(-.05, .05), (-.05, .05), (0, 100)],  
                [(-100, 100), (-100, 100), (-100, 100)]]
    }
    # orientation, centering, and height AND height constraint (1, 2, 3)
    TSR_123_config = {
        'degrees': False,
        "T0_w":  [0.62, -0.6324, 0.15, np.pi/2, -np.pi/2, np.pi/2],
        "Tw_e": [0, 0, 0, 0, 0, 0],
        "Bw": [[(-.05, .05), (-.05, .05), (0, 100)],  
                [(-.1, .1), (-.1, .1), (-.1, .1)]]
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
    script_logger.info("Creating Concept Constrained LfD Sequetial Pose Distribution Model")

    with open(LFD_MODEL_FILEPATH, "r") as f:
        serialized_data = json.load(f)
    lfd_config = serialized_data["config"]
    intermediate_trajectories = serialized_data["intermediate_trajectories"]
    keyframes = OrderedDict(sorted(serialized_data["keyframes"].items(), key=lambda t: int(t[0])))

    for _ in range(0, TRIALS):
        eval_trial = IPDRelaxEvaluationTrial()
        
        # Per trial evaluation data stores.
        TRAJECTORY_SEGMENTS = []
        EVAL_CONSTRAINT_ORDER = []
        IP_GEN_TIMES = []
        IP_GEN_TYPES = []
        IP_TSR_DISTANCES = []
        PLANNING_FAILURE = False
    
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
        #Keyframe bandwidth dictates how heavily biased / overfit out sampling is from our keyframe distribution. In this case we want heavy bias. 
        keyframe_dist = KernelDensityDistribution(bandwidth=.05)
        keyframe_dist.fit(end_data)
        keyframe_space = DistributionSpace(sampler=DistributionSampler(keyframe_dist, fraction_uniform=0, high_confidence_sampling=False), limits=limits)
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
        script_logger.info("Building planning graph")

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
                keyframe_dist = KernelDensityDistribution(bandwidth=.05)
                keyframe_dist.fit(data)
                # We want to fully bias sampling from keyframe distributions.
                keyframe_space = DistributionSpace(sampler=DistributionSampler(keyframe_dist, fraction_uniform=0, high_confidence_sampling=False), limits=limits)

                # Let's create the node and add teh keyframe KDE model as a planning space.
                planning_G.add_nodes_from([keyframe_id], keyframe_space=keyframe_space)

                # get the constraint IDs
                constraint_ids = keyframe_data["applied_constraints"]
                if keyframe_id == 1:
                    constraint_ids = [1]
                # The foliation constraint ids combines both start and end keyframes of the planning segment. In other words, we need to 
                # ensure the start point and ending steering point are in the same foliation, so we utilize the constraints from both keyframes.
                foliation_constraint_ids = list(set(keyframe_data["applied_constraints"] + keyframes[str(upcoming_id)]["applied_constraints"]))

                # we use the current upcoming TSR as the planning TSR...
                planning_G.nodes[keyframe_id]["constraint_ids"] = constraint_ids
                
                # The upcoming Id's unioned constraint ids are used for end point planning to ensure its the upcoming constraint ID
                planning_G.nodes[upcoming_id]["unioned_constraint_ids"] = foliation_constraint_ids
                
                
                # Get the TSR configurations so they can be appended to both the keyframe and the edge between associated with constraint ID combo.
                planning_G.nodes[keyframe_id]['tsr'] = c2tsr_map.get(tuple(sorted(constraint_ids)), unconstrained_TSR)
                planning_config['tsr'] = c2tsr_map.get(tuple(sorted(constraint_ids)), unconstrained_TSR)
                # The upcoming Id's unioned constraint ids are used for end point planning to ensure its the upcoming constraint ID, and this is the TSR configuration
                planning_G.nodes[upcoming_id]['union_tsr'] = c2tsr_map.get(tuple(sorted(foliation_constraint_ids)), unconstrained_TSR)
                # get the foliation model based on the set of constraints
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
                        'bandwidth': .15,
                        'fraction_uniform': .1,
                        'data': inter_trajs_data
                    }
                    planning_config['sampling_bias'] = sampling_bias

                planning_G.add_edge(keyframe_id, upcoming_id)
                # Finally add the planning config to the planning graph edge. 
                planning_G.edges[keyframe_id, upcoming_id]['config'] = planning_config
                script_logger.info("Segment: {} -> {}".format(keyframe_id, upcoming_id))
                script_logger.info("Start point and planning constraints: {}".format(constraint_ids))
                script_logger.info("Start point / planning TSR config: {}".format(planning_config['tsr']))
                script_logger.info("End point constraints: {}".format(planning_G.nodes[upcoming_id]["unioned_constraint_ids"]))
                script_logger.info("End point TSR config: {}".format(planning_G.nodes[upcoming_id]['union_tsr']))
                script_logger.info("")
                # update the upcoming keyframe id with the current id
                upcoming_id = keyframe_id

        script_logger.info("Inserting the starting point into the planning graph: {}".format(start_configuration))
        # Let's insert the starting point to the Planning graph:
        # Copy the base planning config. This will be updated with specfic configurations for this planning segment (tsrs, biasing etc,.)
        planning_config = copy.deepcopy(base_config)
        # We populat ethe "point" attribute of the planning graph node which will indicate that we do not need to sample from this node
        # We also use a basic keyframe space -> TODO: is this necessary?
        planning_G.add_nodes_from([(0, {"point": start_configuration, "keyframe_space": SawyerConfigurationSpace(limits=limits)})])
        # Since this specific task starts off in an orientation constraint, lets use it to plan.
        planning_G.nodes[0]['tsr'] = TSR_1_config
        # let's connect the starting point to the node associated with the starting keyframe
        planning_G.add_edge(0, int(start_keyframe_id))
        keyframe_planning_order.insert(0, 0)
        planning_config['tsr'] = TSR_1_config
        # planning_G.nodes[int(start_keyframe_id)]['tsr'] = TSR_1_config
        # Add the planning config to the planning graph edge. 
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
        rusty_agent_settings_path = str(Path(__file__).parent.absolute()) + "/settings.yaml"

        # Here we use the keyframe planning order, creating a sequential pairing of keyframe ids.
        eval_trial.start_timer("planning_time")
        try:
            for edge in list(zip(keyframe_planning_order, keyframe_planning_order[1:])):
                e1 = edge[0]
                e2 = edge[1]
                script_logger.info("Planning for {} to {}".format(e1, e2))
                edge_data = planning_G.edges[e1, e2]
                # lets ge the planning config from the edge or use the generic base config defined above
                edge_config = edge_data.get('config', base_config)
                if edge_config is None:
                    raise Exception("You need a SimContext planning configuration!")
                # We create a Sim context from the config for planning. 
                sim_context = SawyerBiasedSimContext(configuration=edge_config, setup=False)
                sim_context.setup(sim_overrides={"use_gui": False, "run_parallel": False})
                
                # BIASING CONDITION
                if not biased_planning:
                    planning_state_space = SawyerConfigurationSpace(limits=limits)
                else:
                    planning_state_space = sim_context.get_state_space() # The biased state space for sampling points according to intermediate trajectories.
                    
                EVAL_CONSTRAINT_ORDER.append(planning_G.nodes[e1].get("constraint_ids", None))
                sim = sim_context.get_sim_instance()
                script_logger.info(sim)
                logger = sim_context.get_logger()
                sawyer_robot = sim_context.get_robot()
                svc = sim_context.get_state_validity() # the SVC is the same for all contexts so we will use this one in our planner.
                with DisabledCollisionsContext(sim, [], [], disable_visualization=True):
                    # Create the TSR object
                    e1_tsr_config =  planning_G.nodes[e1].get("tsr", unconstrained_TSR)
                    T0_w = xyzrpy2trans(e1_tsr_config['T0_w'], degrees=e1_tsr_config['degrees'])
                    Tw_e = xyzrpy2trans(e1_tsr_config['Tw_e'], degrees=e1_tsr_config['degrees'])
                    Bw = bounds_matrix(e1_tsr_config['Bw'][0], e1_tsr_config['Bw'][1])
                    # we plan with the current edges first/starting node's tsr and planning space.
                    e1_tsr = TSR(T0_w=T0_w, Tw_e=Tw_e, Bw=Bw)
                    
                    # CANIDIDATE POINT FROM KEYFRAME OR PLANNING SPACE
                    if ip_style == "kf" or ip_style == "optkf":
                        script_logger.info("We're using keyframe planning space for KF-Only or Optimization + KF")
                        keyframe_space_e1 = planning_G.nodes[e1]['keyframe_space']
                    else:
                        keyframe_space_e1 = planning_state_space
                    
                    # generate a starting point, and a steering point, according to constraints (if applicable). 
                    # check if the starting point has generated already:
                    if  planning_G.nodes[e1].get('point', None) is None:
                        foliation_model =  planning_G.nodes[e1].get("foliation_model", None)
                        foliation_value =  planning_G.nodes[e1].get("foliation_value", None)

                        found = False
                        eval_trial.start_timer("steering_point_generation_1")
                        while not found:
                            # we want the candidate sample to be from the foliation of the model
                            if foliation_model is not None:
                                sample_from_foliation = False
                                # so we perform a rejection sampling step
                                while not sample_from_foliation:
                                    raw_sample = keyframe_space_e1.sample()
                                    canididate_sample = []
                                    for value in raw_sample:
                                        canididate_sample.append(wrap_to_interval(value))
                                    if foliation_model.predict(np.array([canididate_sample])) == foliation_value:
                                        sample_from_foliation = True
                            else:
                                raw_sample = keyframe_space_e1.sample()
                                canididate_sample = []
                                for value in raw_sample:
                                    canididate_sample.append(wrap_to_interval(value))
                            err, deltas = distance_to_TSR_config(sawyer_robot, canididate_sample, e1_tsr)
                            constraint_list = planning_G.nodes[e1].get("constraint_ids", None)
                            # If there are no constraints, we directly use the sampeld point. Thanks LfD!
                            if constraint_list is None or constraint_list == []:
                                if svc.validate(canididate_sample):
                                    start = canididate_sample
                                    planning_G.nodes[e1]['point'] = start
                                    script_logger.info("No constraints, using LfD model sampled point!")
                                    script_logger.info("{}".format(start))
                                    found = True
                                    IP_GEN_TYPES.append("direct")
                            # If the sample is already constraint compliant, no need to perform omega optimization. Thanks LfD!
                            elif err < OMEGA_TSR_EPSILON and svc.validate(canididate_sample):
                                    start = canididate_sample
                                    planning_G.nodes[e1]['point'] = start
                                    script_logger.info("Sampled start point TSR compliant for constraints: {}! {} {}".format(constraint_list, err, deltas))
                                    script_logger.info("{}".format(start))
                                    found = True
                                    IP_GEN_TYPES.append("direct")
                            elif ip_style == "optkf" or ip_style == "opt":
                                script_logger.info("Optimization condition indicated, performing Omega Optimization!")
                                if svc.validate(canididate_sample):
                                    # We create an Agent used for OmegaOptimization from planning_core_rust.
                                    rusty_sawyer_robot = Agent(rusty_agent_settings_path, False, False)
                                    # To assist in optimization, we seed the optimizaition with a point generated using inverse kinematics based on the ideal TSR point. 
                                    # seed_start = sawyer_robot.solve_inverse_kinematics(planning_tsr_config["T0_w"][0:3], planning_tsr_config["T0_w"][3:])
                                    # We update the optimization variables with the seed start and the current TSR used for optimization.
                                    rusty_sawyer_robot.update_xopt(canididate_sample)
                                    rusty_sawyer_robot.update_planning_tsr(e1_tsr_config['T0_w'], e1_tsr_config['Tw_e'], e1_tsr_config['Bw'][0] +  e1_tsr_config['Bw'][1])
                                    # The optimization is based on CollisionIK which maintains feasibility with the starting seed start. This feasibility might aid in the optimization staying reasonably close to the ideal TSR sample.
                                    for _ in range(0, 500):
                                        # The sample we are optimizing is passed as an argument to omega_optimize. This feeds the optimization call to bias staying close to this sample. 
                                        q_constrained = rusty_sawyer_robot.omega_optimize(canididate_sample).data
                                    if any([np.isnan(val) for val in q_constrained]):
                                        continue
                                    normalized_q_constrained = []
                                    for value in q_constrained:
                                        normalized_q_constrained.append(
                                            wrap_to_interval(value))
                                    # If there is a foliation model, then we must perform rejection sampling until the projected sample is classified 
                                    # to the node's foliation value
                                    err, deltas = distance_to_TSR_config(sawyer_robot, q_constrained, e1_tsr)
                                    # We do one last check to ensure the optimized sample is TSR compliant.
                                    if err < OMEGA_TSR_EPSILON:
                                        # If it is, we then check to see if the sample classifies into the learned foliation/disjoint set choice/ID of the model which was learned from human demonstration data.
                                        if foliation_model is not None:
                                            # This is the rejection sampling step to enforce the foliation choice
                                            if foliation_model.predict(np.array([normalized_q_constrained])) != foliation_value:
                                                continue
                                        if svc.validate(normalized_q_constrained):
                                            start = normalized_q_constrained
                                            # We've generated a point so lets use it moving forward for all other planning segments. 
                                            planning_G.nodes[e1]['point'] = start
                                            script_logger.info("Original point that was optimized: {}".format(canididate_sample))
                                            script_logger.info("Omega Optimized Start Point for constraints: {}.".format(constraint_list))
                                            script_logger.info("Omega Optimized Point TSR Errors: {} {}".format(err, deltas))
                                            script_logger.info("{}", start)
                                            found = True
                                            IP_GEN_TYPES.append("optimization")
                                        else: continue
                                    else: continue
                                else: continue
                            elif ip_style == "kf":
                                start = canididate_sample
                                planning_G.nodes[e1]['point'] = start
                                script_logger.info("KF-only condition, using KF sampled point!")
                                script_logger.info("{}".format(start))
                                found = True
                                IP_GEN_TYPES.append("direct")
                            else: continue
                        IP_GEN_TIMES.append(eval_trial.end_timer("steering_point_generation_1"))
                        # Evaluate TSR distance for each point.
                        err, deltas = distance_to_TSR_config(sawyer_robot, start, e1_tsr)
                        if e1_tsr is not None:
                            IP_TSR_DISTANCES.append(err)
                        else:
                            IP_TSR_DISTANCES.append(0)
                            # Create a line between the two points. 
                    # If the ending/steering point has been generated from the prior iteration, we use it as our starting point. 
                    else:
                        start = planning_G.nodes[e1]['point']
                        script_logger.info("Reusing previously acquired point: {}".format(start))

                    if  planning_G.nodes[e2].get('point', None) is None:
                        keyframe_space_e2 =  planning_G.nodes[e2]['keyframe_space']
                        e2_tsr_config =  planning_G.nodes[e2].get("union_tsr", unconstrained_TSR)
                        T0_w2 = xyzrpy2trans(e2_tsr_config['T0_w'], degrees=e2_tsr_config['degrees'])
                        Tw_e2 = xyzrpy2trans(e2_tsr_config['Tw_e'], degrees=e2_tsr_config['degrees'])
                        Bw2 = bounds_matrix(e2_tsr_config['Bw'][0], e2_tsr_config['Bw'][1])
                        e2_tsr = TSR(T0_w=T0_w2, Tw_e=Tw_e2, Bw=Bw2)
                        
                        print("Constraints {}: {}".format(e2, planning_G.nodes[e2].get("unioned_constraint_ids", [])))
                        foliation_model =  planning_G.nodes[e2].get("foliation_model", None)
                        foliation_value =  planning_G.nodes[e2].get("foliation_value", None)
                        
                        found = False
                        eval_trial.start_timer("steering_point_generation_2")
                        while not found:
                            if foliation_model is not None:
                                sample_from_foliation = False
                                while not sample_from_foliation:
                                    raw_sample = keyframe_space_e2.sample()
                                    canidate_sample = []
                                    for value in raw_sample:
                                        canidate_sample.append(wrap_to_interval(value))
                                    if foliation_model.predict(np.array([canidate_sample])) == foliation_value:
                                        sample_from_foliation = True
                            else:
                                raw_sample = keyframe_space_e2.sample()
                                canidate_sample = []
                                for value in raw_sample:
                                    canidate_sample.append(wrap_to_interval(value))
                            err, deltas = distance_to_TSR_config(sawyer_robot, canidate_sample, e2_tsr)
                            constraint_list = planning_G.nodes[e2].get("unioned_constraint_ids", None)
                            if constraint_list is None or constraint_list == []:
                                if svc.validate(canidate_sample):
                                    end = canidate_sample
                                    planning_G.nodes[e2]['point'] = end
                                    script_logger.info("No constraints so using LfD model sampled point!")
                                    script_logger.info("{}".format(end))
                                    found = True
                                    IP_GEN_TYPES.append("direct")

                            elif err < OMEGA_TSR_EPSILON and svc.validate(canidate_sample):
                                    end = canidate_sample
                                    planning_G.nodes[e2]['point'] = end
                                    script_logger.info("Sampled end point TSR compliant for constraints: {}! {} {}".format(constraint_list, err, deltas))
                                    script_logger.info("{}".format(end))
                                    found = True
                                    IP_GEN_TYPES.append("direct")

                            elif ip_style == "optkf" or ip_style == "opt":
                                script_logger.info("Optimization condition indicated, performing Omega Optimization!")
                                if svc.validate(canidate_sample):
                                    rusty_sawyer_robot = Agent(rusty_agent_settings_path, False, False)
                                    # seed_start = sawyer_robot.solve_inverse_kinematics(tsr_config["T0_w"][0:3], tsr_config["T0_w"][3:])
                                    rusty_sawyer_robot.update_xopt(canidate_sample)
                                    rusty_sawyer_robot.update_planning_tsr(e2_tsr_config['T0_w'], e2_tsr_config['Tw_e'], e2_tsr_config['Bw'][0] + e2_tsr_config['Bw'][1])
                                    # we use the planning TSR used for the constrained planner as a secondary target.
                                    for _ in range(0, 1000):
                                        q_constrained = rusty_sawyer_robot.omega_optimize(canidate_sample).data
                                    normalized_q_constrained = []
                                    if any([np.isnan(val) for val in q_constrained]):
                                        continue
                                    for value in q_constrained:
                                        normalized_q_constrained.append(
                                            wrap_to_interval(value))
                                    err, deltas = distance_to_TSR_config(sawyer_robot, normalized_q_constrained, e2_tsr)
                                    if err < OMEGA_TSR_EPSILON and q_constrained is not None:
                                        if foliation_model is not None:
                                            # This is the rejection sampling step to enforce the foliation choice
                                            if foliation_model.predict(np.array([normalized_q_constrained])) != foliation_value:
                                                continue
                                        if svc.validate(normalized_q_constrained):
                                            end = normalized_q_constrained
                                            # We've generated a point so lets use it moving forward for all other planning segments. 
                                            planning_G.nodes[e2]['point'] = end
                                            script_logger.info("Original point that was optimized: {}".format(canidate_sample))
                                            script_logger.info("Omega Optimized End Point for constraints: {}.".format(constraint_list))
                                            script_logger.info("Omega Optimized Point TSR Errors: {} {}".format(err, deltas))
                                            script_logger.info("{}".format(end))
                                            found = True
                                            IP_GEN_TYPES.append("optimization")
                                        else: continue
                                    else: continue
                                else: continue
                            elif ip_style == "kf":
                                end = canididate_sample
                                planning_G.nodes[e2]['point'] = end
                                script_logger.info("KF-only condition, using KF sampled point!")
                                script_logger.info("{}".format(end))
                                found = True
                                IP_GEN_TYPES.append("direct")
                            else: continue
                        IP_GEN_TIMES.append(eval_trial.end_timer("steering_point_generation_2"))
                        # Evaluate TSR distance for each point.
                        err, deltas = distance_to_TSR_config(sawyer_robot, end, e2_tsr)
                        if e2_tsr is not None:
                            IP_TSR_DISTANCES.append(err)
                        else:
                            IP_TSR_DISTANCES.append(0)
                            # Create a line between the two points. 
                    else:
                        end = planning_G.nodes[e2]['point']
                        script_logger.info("Reusing previously acquired point")
                        script_logger.info("{}".format(end))
                        

                    print("\n\nSTART AND END\n")
                    print(start, end)
                    script_logger.info("Planning start: {}".format(start))
                    script_logger.info("Planning end: {}".format(end))
                    print(np.linalg.norm(np.array(start) - np.array(end)))
                    print("\n\n")
                    if np.linalg.norm(np.array(start) - np.array(end)) > .1:
                        ###########
                        # CBiRRT2 #
                        ###########
                        edge_tsr_config = edge_config.get('tsr', unconstrained_TSR)
                        print(edge_tsr_config)
                        if e1 == 0:
                            time.sleep(10)
                        T0_w = xyzrpy2trans(edge_tsr_config['T0_w'], degrees=edge_tsr_config['degrees'])
                        Tw_e = xyzrpy2trans(edge_tsr_config['Tw_e'], degrees=edge_tsr_config['degrees'])
                        Bw = bounds_matrix(edge_tsr_config['Bw'][0], edge_tsr_config['Bw'][1])
                        planning_tsr = TSR(T0_w=T0_w, Tw_e=Tw_e, Bw=Bw)
                        # Use parametric linear interpolation with 10 steps between points.
                        interp = partial(parametric_lerp, steps=20)
                        # See params for CBiRRT2 specific parameters 
                        cbirrt = CBiRRT2(sawyer_robot, planning_state_space, svc, interp, params={'smooth_path': SMOOTH, 'smoothing_time': SMOOTHING_TIME, 'epsilon': PLANNING_TSR_EPSILON, 'q_step': Q_STEP, 'e_step': E_STEP, 'iters': MAX_ITERS})
                        logger.info("Planning....")
                        print("Start, end: ", start, end)
                        logger.info("Constraints: {}".format(planning_G.nodes[e1].get('constraint_ids', None)))
                        script_logger.info("Planning with constraints: {}".format(planning_G.nodes[e1].get('constraint_ids', None)))
                        print(edge_tsr_config)
                        plan = cbirrt.plan(planning_tsr, np.array(start), np.array(end))
                        path = cbirrt.get_path(plan)
                        if len(path) == 0:
                            logger.info("Planning failed....")
                            PLANNING_FAILURE = True
                        logger.info("Plan found....")
                        script_logger.info("Plan found for {} to {}".format(e1, e2))
                    else:
                        interp = partial(parametric_lerp, steps=20)
                        # sometimes the start point is really, really close to the a keyframe so we just inerpolate, since really close points are challenging the CBiRRT2 given the growth parameters
                        path = [list(val) for val in interp(np.array(start), np.array(end))]

                        # splining uses numpy so needs to be converted
                        logger.info("Length of interpolated path: {}".format(len(path)))
                    
                    TRAJECTORY_SEGMENTS.append(path)
                    final_path = final_path + path
                sim_context.disconnect()  
        except PlanningTimeoutException:
            print("PLANNING TIMEOUT! PLANNING FAILURE!")
            PLANNING_FAILURE = True
            eval_trial.notes = "Planning timeout failure."
        except MaxItersException:
            print("MAX ITERS REACHED. PLANNING FAILURE!")
            PLANNING_FAILURE = True
            eval_trial.notes = "Planning max iters failure."                    
        
        
        if not PLANNING_FAILURE:
            eval_trial.planning_time = eval_trial.end_timer("planning_time")
            # splining uses numpy so needs to be converted
            planning_path = [np.array(p) for p in final_path]
            # Create a MinJerk spline trajectory using JointTrajectoryCurve and execute
            jtc = JointTrajectoryCurve()
            traj = jtc.generate_trajectory(planning_path, move_time=10)
            
            # Build map of actual TSR objects for evaluation
            cs2tsr_object_map = {}
            for key, config in c2tsr_map.items():
                T0_w = xyzrpy2trans(config['T0_w'], degrees=config['degrees'])
                Tw_e = xyzrpy2trans(config['Tw_e'], degrees=config['degrees'])
                Bw = bounds_matrix(config['Bw'][0], config['Bw'][1])
                cs2tsr_object_map[key] = TSR(T0_w=T0_w, Tw_e=Tw_e, Bw=Bw)
            
            # Update trial evaluation data.
            eval_trial.path_length = eval_trial.eval_path_length(trajectory)
            eval_trial.success = eval_trial.eval_success(trajectory, goal, EPSILON)
            eval_trial.a2s_distance = eval_trial.eval_a2s([p[:2] for p in trajectory], [p[:2] for p in gold_demo_traj])
            eval_trial.a2f_percentage = eval_trial.eval_a2f(TRAJECTORY_SEGMENTS, cs2tsr_object_map, EVAL_CONSTRAINT_ORDER)
            eval_trial.ip_gen_times = IP_GEN_TIMES
            eval_trial.ip_gen_types = IP_GEN_TYPES
            eval_trial.ip_tsr_distances = IP_TSR_DISTANCES
            eval_trial.trajectory = traj
            evaluation.add_trial(eval_trial)
                                        
            if VISUALIZE_EXECUTION:
                sim_context = SawyerBiasedSimContext(configuration=base_config, setup=False)
                sim_context.setup(sim_overrides={"use_gui": True, "run_parallel": False})
                sim = sim_context.get_sim_instance()
                logger = sim_context.get_logger()
                sawyer_robot = sim_context.get_robot()
                svc = sim_context.get_state_validity()
                interp_fn = partial(parametric_lerp, steps=20)

        
                sawyer_robot.set_joint_state(start_configuration)
            
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
                    sim_context.disconnect()
        else:
            # Update trial evaluation data with failure-style data. Many defaults are already set.
            eval_trial.success = "X"
            evaluation.add_trial(eval_trial)               

    evaluation.export()


