import os
import sys
from functools import partial
import time
import json

import pybullet as p
if os.environ.get('ROS_DISTRO'):
    import rospy
import numpy as np

from cairo_simulator.core.sim_context import SawyerTSRSimContext
from cairo_simulator.core.utils import ASSETS_PATH

from cairo_planning.collisions import DisabledCollisionsContext
from cairo_planning.local.interpolation import parametric_lerp
from cairo_planning.local.curve import JointTrajectoryCurve
from cairo_planning.planners import CBiRRT2
from cairo_planning.geometric.state_space import SawyerConfigurationSpace




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
        

    start = [
        0.673578125,
        -0.2995908203125,
        -0.21482421875,
        1.4868740234375,
        0.53829296875,
        0.4117080078125,
        -1.2169501953125]

    goal = [
        -1.3020732421875,
        -0.44705859375,
        0.6508818359375,
        1.5064189453125,
        -0.889978515625,
        0.8245869140625,
        -1.6826474609375]

    sim_context = SawyerTSRSimContext(configuration=config)
    sim = sim_context.get_sim_instance()
    logger = sim_context.get_logger()
    _ = sim_context.get_state_space()
    sawyer_robot = sim_context.get_robot()
    # _ = sawyer_robot.get_simulator_id()
    tsr = sim_context.get_tsr()
    _ = sim_context.get_sim_objects(['Ground'])[0]
    svc = sim_context.get_state_validity()

    state_space = SawyerConfigurationSpace()
    
    sawyer_robot.set_joint_state(start)
    time.sleep(5)

    control = 'g'

    while control != 'q':
        

        with DisabledCollisionsContext(sim, [], [], disable_visualization=True):
            #######
            # LazyPRM #
            #######
            # Use parametric linear interpolation with 5 steps between points.
            interp = partial(parametric_lerp, steps=10)
            # See params for PRM specific parameters
            cbirrt = CBiRRT2(sawyer_robot, state_space, svc, interp, params={'smooth_path': True, 'smoothing_time': 10, 'q_step': .1, 'e_step': .25, 'iters': 20000})
            logger.info("Planning....")
            plan = cbirrt.plan(tsr, np.array(start), np.array(goal))
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
            traj = jtc.generate_trajectory(path, move_time=5)
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
