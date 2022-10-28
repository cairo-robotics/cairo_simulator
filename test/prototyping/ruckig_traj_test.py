import sys
import os
import time
import json
from functools import partial

import numpy as np
import pybullet as p
if os.environ.get('ROS_DISTRO'):
    import rospy

from cairo_simulator.core.sim_context import SawyerSimContext
from cairo_simulator.core.simulator import SimObject
from cairo_simulator.core.primitives import create_box
from cairo_planning.core.planning_context import SawyerPlanningContext
from cairo_planning.collisions import DisabledCollisionsContext
from cairo_planning.local.interpolation import parametric_lerp
from cairo_planning.local.curve import JointTrajectoryCurve
from cairo_planning.planners import LazyPRM


from copy import copy


from ruckig import InputParameter, OutputParameter, Result, Ruckig, Synchronization, Interface, DurationDiscretization


import matplotlib.pyplot as plt
import numpy as np

import os
import sys
from functools import partial
import time

import pybullet as p
if os.environ.get('ROS_DISTRO'):
    import rospy
import numpy as np

from cairo_simulator.core.sim_context import SawyerSimContext

from cairo_planning.collisions import DisabledCollisionsContext
from cairo_planning.local.interpolation import parametric_lerp
from cairo_planning.local.curve import JointTrajectoryCurve
from cairo_planning.planners import LazyPRM

from ruckig import OutputParameter, Result

def walk_through_trajectory(otg, inp, intermediate_targets):
    t_list, out_list = [], []
    out = OutputParameter(inp.degrees_of_freedom)

    time_offset = 0.0
    for inp.target_position, inp.target_velocity, inp.target_acceleration in intermediate_targets:
        res = Result.Working
        while res == Result.Working:
            res = otg.update(inp, out)

            inp.current_position = out.new_position
            inp.current_velocity = out.new_velocity
            inp.current_acceleration = out.new_acceleration

            t_list.append(time_offset + out.time)
            out_list.append(copy(out))

        time_offset += out.trajectory.duration

    return t_list, out_list, time_offset



  
def main():
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

    config = {}
    config["sim_objects"] = [
        {
            "object_name": "Ground",
            "model_file_or_sim_id": "plane.urdf",
            "position": [0, 0, 0]
        },
        {
            "object_name": "sphere",
            "model_file_or_sim_id": 'sphere2.urdf',
            "position": [1.0, -.3, .6],
            "orientation":  [0, 0, 1.5708],
            "fixed_base": 1    
        }
        # {
        #     "object_name": "Table",
        #     "model_file_or_sim_id": ASSETS_PATH + 'table.sdf',
        #     "position": [.8, -.6, .6],
        #     "orientation":  [0, 0, 1.5708],
        #     "fixed_base": 1
        # }
    ]
    sim_context = SawyerSimContext(config)
    sim = sim_context.get_sim_instance()
    logger = sim_context.get_logger()
    state_space = sim_context.get_state_space()
    sawyer_robot = sim_context.get_robot()
    svc = sim_context.get_state_validity()
    interp_fn = partial(parametric_lerp, steps=20)
    _ = sawyer_robot.get_simulator_id()
    _ = sim_context.get_sim_objects(['Ground'])[0]

    start = [0, 0, 0, 0, 0, 0, -np.pi/2]

    goal = [-1.9622245072067646, 0.8439858364277937, 1.3628459180018329, -
            0.2383928041974519, -2.7327884695211555, -2.2177502341009134, -0.08992133311928363]

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
        prm = LazyPRM(state_space, svc, interp_fn, params={
                'n_samples': 5000, 'k': 8, 'ball_radius': 2.5})
        logger.info("Planning....")
        plan = prm.plan(np.array(start), np.array(goal))
        # get_path() reuses the interp function to get the path between vertices of a successful plan
        path = prm.get_path(plan)
    if len(path) == 0:
        logger.info("Planning failed....")
        sys.exit(1)
    logger.info("Plan found....")

    # splinging uses numpy so needs to be converted
    path = [np.array(p) for p in path]
    logger.info("Length of path: {}".format(len(path)))

    sawyer_robot.move_to_joint_pos(start)
    time.sleep(3)
    while sawyer_robot.check_if_at_position(start, 0.5) is False:
        time.sleep(0.1)
        sim.step()

    if len(path) > 0:
        # Create a MinJerk spline trajectory using JointTrajectoryCurve and execute
        jtc = JointTrajectoryCurve()
        traj = jtc.generate_trajectory(path, move_time=10)
        kinematics_traj = [point[1:4] for point in traj]
        print(len(kinematics_traj))
        inp = InputParameter(7)
        inp.current_position = start
        inp.current_velocity = [0, 0, 0, 0, 0, 0, 0]
        inp.current_acceleration = [0, 0, 0, 0, 0, 0, 0]
        inp.target_position = goal
        inp.target_velocity = [0, 0, 0, 0, 0, 0, 0]
        inp.target_acceleration = [0, 0, 0, 0, 0, 0, 0]
        # inp.max_velocity = [.1] * 7
        inp.max_velocity = [.88, .678, .996, .996, 1.776, 1.776, 2.316]
        inp.max_acceleration = [3.5, 2.5, 5, 5, 5, 5, 5]
        inp.max_jerk = [10] * 7

        intermediate_targets = kinematics_traj[1:]

        otg = Ruckig(7, 2)
    
        t_list, out_list, duration = walk_through_trajectory(otg, inp, intermediate_targets)
        output_pos = [out_pt.new_position for out_pt in out_list]
        output_vel = [out_pt.new_velocity for out_pt in out_list]
        output_acel = [out_pt.new_acceleration for out_pt in out_list]
        smooth_traj = list(zip([time for time in t_list], output_pos, output_vel, output_acel))
        traj_data = {}
        traj_data["trajectory"] = []
        for point in smooth_traj:
            traj_point = {
                "time": point[0],
                "position": point[1],
                "velocity": point[2],
                "acceleration": point[3]
            }
            traj_data['trajectory'].append(traj_point)
        with open('traj_w_id.json', 'w') as f:
            json.dump(traj_data, f)
        sawyer_robot.execute_trajectory(list(zip( [point[0] for point in smooth_traj],  [point[1] for point in smooth_traj])))
        try:
            while True:
                sim.step()
        except KeyboardInterrupt:
            sys.exit(0)
    else:
        logger.err("No path found.")

   

    


if __name__ == "__main__":
    main()
