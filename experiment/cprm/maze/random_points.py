import random
import time
import copy
import numpy as np
from cairo_simulator.core.sim_context import SawyerSimContext
from cairo_simulator.core.utils import ASSETS_PATH
from cairo_planning.collisions import DisabledCollisionsContext
from cairo_planning.geometric.transformation import xyzrpy2trans, bounds_matrix
from cairo_planning.constraints.projection import project_config
from cairo_planning.geometric.utils import wrap_to_interval
from cairo_planning.geometric.tsr import TSR

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
            'urdf_file': ASSETS_PATH + 'sawyer_description/urdf/sawyer_static_blockcombine.urdf',
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
            "type": "box",
            "primitive_configs": {"w": .015, "l": .35, "h": .08},
            "sim_object_configs": 
                {
                    "object_name": "center_wall",
                    "position": [.62, 0, .64],
                    "orientation":  [0, 0, 0],
                    "fixed_base": 1    
                }
        },
        {
            "type": "box",
            "primitive_configs": {"w": .015, "l": .2, "h": .08},
            "sim_object_configs": 
                {
                    "object_name": "right_outer_bend",
                    "position": [.74, .25, .64],
                    "orientation":  [0, 0, -np.pi/6],
                    "fixed_base": 1    
                }
        },
        {
            "type": "box",
            "primitive_configs": {"w": .015, "l": .2, "h": .08},
            "sim_object_configs": 
                {
                    "object_name": "right_outer_bend2",
                    "position": [.57, .25, .64],
                    "orientation":  [0, 0, np.pi/6],
                    "fixed_base": 1    
                }
        },
         {
            "type": "box",
            "primitive_configs": {"w": .015, "l": .10, "h": .08},
            "sim_object_configs": 
                {
                    "object_name": "right_inner_bend",
                    "position": [.68, .29, .64],
                    "orientation":  [0, 0, -np.pi/6],
                    "fixed_base": 1    
                }
        },
        {
            "type": "box",
            "primitive_configs": {"w": .015, "l": .10, "h": .08},
            "sim_object_configs": 
                {
                    "object_name": "right_inner_bend2",
                    "position": [.64, .29, .64],
                    "orientation":  [0, 0, np.pi/6],
                    "fixed_base": 1    
                }
        },
        {
            "type": "box",
            "primitive_configs": {"w": .015, "l": .2, "h": .08},
            "sim_object_configs": 
                {
                    "object_name": "left_outer_bend",
                    "position": [.74, -.25, .64],
                    "orientation":  [0, 0, np.pi/6],
                    "fixed_base": 1    
                }
        },
        {
            "type": "box",
            "primitive_configs": {"w": .015, "l": .2, "h": .08},
            "sim_object_configs": 
                {
                    "object_name": "left_outer_bend2",
                    "position": [.57, -.25, .64],
                    "orientation":  [0, 0, -np.pi/6],
                    "fixed_base": 1    
                }
        },
         {
            "type": "box",
            "primitive_configs": {"w": .015, "l": .10, "h": .08},
            "sim_object_configs": 
                {
                    "object_name": "left_inner_bend",
                    "position": [.68, -.29, .64],
                    "orientation":  [0, 0, np.pi/6],
                    "fixed_base": 1    
                }
        },
        {
            "type": "box",
            "primitive_configs": {"w": .015, "l": .10, "h": .08},
            "sim_object_configs": 
                {
                    "object_name": "left_inner_bend2",
                    "position": [.64, -.29, .64],
                    "orientation":  [0, 0, -np.pi/6],
                    "fixed_base": 1    
                }
        },
        {
            "type": "box",
            "primitive_configs": {"w": .015, "l": .35, "h": .08},
            "sim_object_configs": 
                {
                    "object_name": "center_wall2",
                    "position": [.69, 0, .64],
                    "orientation":  [0, 0, 0],
                    "fixed_base": 1    
                }
        },
        ]
            
    config['tsr'] = {
            'degrees': False,
            'epsilon': .05,
            'e_step': .25,
            "T0_w": [.676, 0, .84, 0, 0, 0],
            "Tw_e": [0, 0, 0, np.pi, 0, np.pi/2],
            "Bw": [[(-.1, .1), (-100, 100), (0, .005)],  
                    [(-.001, .001), (-.001, .001), (-.001, .001)]]
    }

    start = [0.4512693515323636, 0.578072751043309, -1.7085853204387587,
             0.596159706024823, 1.9871449177039127, 1.2134687707559257, -1.569380122838989]
    goal = [-0.5684057726305594, 0.5583954509945905, -1.5247855621059458,
            0.7754717976826726, 1.901730705121558, 1.135705090297649, -2.8032179515916686]


    sim_context = SawyerSimContext(config)
    sim = sim_context.get_sim_instance()
    scs = sim_context.get_state_space()
    svc = sim_context.get_state_validity()
    sawyer_robot = sim_context.get_robot()

        
    sawyer_robot.set_joint_state(start)
    print("Basis start position")
    time.sleep(3)

    sawyer_robot.set_joint_state(goal)
    print("Basis goal position")
    time.sleep(3)

    start_world_pose, start_local_pose = sawyer_robot.solve_forward_kinematics(
        start)
    goal_world_pose, goal_local_pose = sawyer_robot.solve_forward_kinematics(
        goal)

    random_start_configurations = []
    for _ in range(0, 10):
        found = False
        while not found:
            delta_x = random.uniform(-.3, .3)
            delta_y = random.uniform(0, .1)
            new_position = copy.deepcopy([copy.deepcopy(start_world_pose[0][0]) + delta_x, copy.deepcopy(start_world_pose[0][1]) + delta_y, copy.deepcopy(start_world_pose[0][2])])
            
            new_config = sawyer_robot.solve_inverse_kinematics(new_position, copy.deepcopy(start_world_pose[1]))
            print(new_config)
                # Utilizes RPY convention
            config['tsr'] = {
                'degrees': False,
                'epsilon': .05,
                'e_step': .25,
                "T0_w": [new_position[0], new_position[1], .84, 0, 0, 0],
                "Tw_e": [0, 0, 0, np.pi, 0, np.pi/2],
                "Bw": [[(-.01, .01), (-.01, .01), (0, .01)],  
                        [(-.01, .01), (-.01, .01), (-.01, .01)]]
            }
            T0_w = xyzrpy2trans(config["tsr"]["T0_w"], degrees=False)

            # Utilizes RPY convention
            Tw_e = xyzrpy2trans(config["tsr"]["Tw_e"], degrees=False)
            
            # Utilizes RPY convention
            Bw = bounds_matrix(config["tsr"]["Bw"][0],  # allow some tolerance in the z and y and only positve in x
                            config["tsr"]["Bw"][1])  # any rotation about z, with limited rotation about x, and y.
            tsr = TSR(T0_w=T0_w, Tw_e=Tw_e, Bw=Bw,
                    manipindex=0, bodyandlink=16)

            # Disabled collisions during planning with certain eclusions in place.
            with DisabledCollisionsContext(sim, [], []):
                if svc.validate(new_config):
                    q_constrained = project_config(sawyer_robot, tsr, np.array(
                    new_config), np.array(new_config), epsilon=.05, e_step=.25)
                    print(q_constrained)
                    if q_constrained is not None:
                        random_start_configurations.append(
                            list(q_constrained))
                        found = True

    random_goal_configurations = []
    for _ in range(0, 10):
        found = False
        while not found:
            delta_x = random.uniform(-.3, .3)
            delta_y = random.uniform(-.1, 0)
            new_position = copy.deepcopy([copy.deepcopy(goal_world_pose[0][0]) + delta_x, copy.deepcopy(goal_world_pose[0][1]) + delta_y, copy.deepcopy(start_world_pose[0][2])])
            
            new_config = sawyer_robot.solve_inverse_kinematics(new_position, copy.deepcopy(goal_world_pose[1]))
            print(new_config)
                # Utilizes RPY convention
            config['tsr'] = {
                'degrees': False,
                'epsilon': .05,
                'e_step': .25,
                "T0_w": [new_position[0], new_position[1], .84, 0, 0, 0],
                "Tw_e": [0, 0, 0, np.pi, 0, np.pi/2],
                "Bw": [[(-.01, .01), (-.01, .01), (0, .01)],  
                        [(-.01, .01), (-.01, .01), (-.01, .01)]]
            }
            T0_w = xyzrpy2trans(config["tsr"]["T0_w"], degrees=False)

            # Utilizes RPY convention
            Tw_e = xyzrpy2trans(config["tsr"]["Tw_e"], degrees=False)
            
            # Utilizes RPY convention
            Bw = bounds_matrix(config["tsr"]["Bw"][0],  # allow some tolerance in the z and y and only positve in x
                            config["tsr"]["Bw"][1])  # any rotation about z, with limited rotation about x, and y.
            tsr = TSR(T0_w=T0_w, Tw_e=Tw_e, Bw=Bw,
                    manipindex=0, bodyandlink=16)

            # Disabled collisions during planning with certain eclusions in place.
            with DisabledCollisionsContext(sim, [], []):
                if svc.validate(new_config):
                    q_constrained = project_config(sawyer_robot, tsr, np.array(
                    new_config), np.array(new_config), epsilon=.05, e_step=.25)
                    print(q_constrained)
                    if q_constrained is not None:
                        random_goal_configurations.append(
                            list(q_constrained))
                        found = True

    print(random_start_configurations)
    print(random_goal_configurations)

    for point in random_start_configurations:
        sawyer_robot.set_joint_state(point)
        time.sleep(1)
    for point in random_goal_configurations:
        sawyer_robot.set_joint_state(point)
        time.sleep(1)


if __name__ == "__main__":
    main()
