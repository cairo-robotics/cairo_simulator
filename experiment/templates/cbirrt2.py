import os
import sys
from functools import partial
import time

import pybullet as p
if os.environ.get('ROS_DISTRO'):
    import rospy
import numpy as np

from cairo_simulator.core.sim_context import SawyerCPRMSimContext
from cairo_simulator.core.simulator import SimObject
from cairo_simulator.core.primitives import create_box
from cairo_simulator.core.utils import ASSETS_PATH

from cairo_planning.collisions import DisabledCollisionsContext
from cairo_planning.local.interpolation import parametric_lerp
from cairo_planning.local.curve import JointTrajectoryCurve
from cairo_planning.planners import CBiRRT2
from cairo_planning.geometric.state_space import SawyerConfigurationSpace
from cairo_planning.geometric.transformation import xyzrpy2trans, bounds_matrix, quat2rpy
from cairo_planning.geometric.tsr import TSR
from cairo_planning.geometric.utils import geodesic_distance, wrap_to_interval

from cairo_planning.core.serialization import load_model

def main():

    number_of_planning_attempts = 50
    planning_times = []
    execution_times = []
    planning_failures = 0
    execution_failures = 0
    start = [0, 0, 0, 0, 0, 0, -np.pi/2]
    goal = [-1.9622245072067646, 0.8439858364277937, 1.3628459180018329, -
            0.2383928041974519, -2.7327884695211555, -2.2177502341009134, -0.08992133311928363]

     # Reload the samples and configuration
    directory = os.path.join(os.path.dirname(
        os.path.abspath(__file__)), "serialization_data/2021-08-17T16-25-40")
    config, _, _ = load_model(directory)
    config["sim_objects"] = [
        {
            "object_name": "Ground",
            "model_file_or_sim_id": "plane.urdf",
            "position": [0, 0, 0]
        },
        {
            "object_name": "sphere1",
            "model_file_or_sim_id": 'sphere2.urdf',
            "position": [1.0, -.3, .6],
            "orientation":  [0, 0, 1.5708],
            "fixed_base": 1
        },
        {
            "object_name": "sphere2",
            "model_file_or_sim_id": 'sphere2.urdf',
            "position": [1.0, -.3, 1.65],
            "orientation":  [0, 0, 1.5708],
            "fixed_base": 1
        }
    ]
    sim_context = SawyerCPRMSimContext(config)
    sim = sim_context.get_sim_instance()
    logger = sim_context.get_logger()
    _ = sim_context.get_state_space()
    sawyer_robot = sim_context.get_robot()
    _ = sawyer_robot.get_simulator_id()
    _ = sim_context.get_sim_objects(['Ground'])[0]
    # box = SimObject('box', create_box(w=.5, l=.5, h=.5), (.7, -0.25, .45), fixed_base=1)

    svc = sim_context.get_state_validity()
    tsr = sim_context.get_tsr()

    for _ in range(0, number_of_planning_attempts):

        # sawyer_robot.move_to_joint_pos(goal)
        # time.sleep(5)
        sawyer_robot.move_to_joint_pos(start)
        time.sleep(5.0)

        planning_space = SawyerConfigurationSpace()
        ptime1 = time.process_time()
        with DisabledCollisionsContext(sim, [], []):
            ###########
            # CBiRRT2 #
            ###########
            # Use parametric linear interpolation with 5 steps between points.
            interp = partial(parametric_lerp, steps=10)
            # See params for PRM specific parameters
            cbirrt = CBiRRT2(sawyer_robot, planning_space, svc,
                             interp, params={'q_step': .48, 'e_step': .25, 'smooth_path': True, 'smoothing_time': 5}, logger=logger)
            logger.info("Planning....")
            plan = cbirrt.plan(tsr, np.array(start), np.array(goal))
            path = cbirrt.get_path(plan)

        if len(path) == 0:
            logger.info("Planning failed....")
            planning_failures += 1
            continue
        # splining uses numpy so needs to be converted
        path = [np.array(p) for p in path]
        # Create a MinJerk spline trajectory using JointTrajectoryCurve and execute
        jtc = JointTrajectoryCurve()
        traj = jtc.generate_trajectory(path, move_time=20)
        ptime2 = time.process_time()
        planning_times.append(ptime2 - ptime1)
        etime1 = time.process_time()
        sawyer_robot.execute_trajectory(traj)
        try:
            while sawyer_robot.check_if_at_position(goal, 0.25) is False:
                sim.step()
                etime_int = time.process_time()
                if etime_int - etime1 > 45:
                    execution_failures += 1
                    break
            if sawyer_robot.check_if_at_position(goal, 0.25) is True:
                etime2 = time.process_time()
                execution_times.append(etime2 - etime1)
            print(planning_times)
            print(execution_times)
            print(planning_failures)
            print(execution_failures)
        except KeyboardInterrupt:
            p.disconnect()
            sys.exit(0)


if __name__ == "__main__":
    main()
