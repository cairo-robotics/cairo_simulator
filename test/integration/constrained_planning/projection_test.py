import os
import sys
import time
import timeit

import pybullet as p
if os.environ.get('ROS_DISTRO'):
    import rospy
import numpy as np

from cairo_simulator.core.context import SawyerSimContext
from cairo_simulator.core.utils import ASSETS_PATH
from cairo_simulator.core.link import get_joint_info_by_name

from cairo_planning.collisions import DisabledCollisionsContext
from cairo_planning.geometric.transformation import xyzrpy2trans, bounds_matrix, quat2rpy
from cairo_planning.constraints.projection import project_config
from cairo_planning.geometric.tsr import TSR
from cairo_planning.geometric.utils import geodesic_distance, wrap_to_interval


def main():

    configuration = {}

    configuration["sim"] = {
        "use_real_time": True,
        "use_gui": True,
        "run_parallel": False
    }

    configuration["logging"] = {
        "handlers": ['logging'],
        "level": "debug"
    }

    configuration["sawyer"] = {
        "robot_name": "sawyer0",
        "position": [0, 0, 0.9],
        "fixed_base": True
    }

    configuration["sim_objects"] = [
        {
            "object_name": "Ground",
            "model_file_or_sim_id": "plane.urdf",
            "position": [0, 0, 0]
        },
        # {
        #     "object_name": "Table",
        #     "model_file_or_sim_id": ASSETS_PATH + 'table.sdf',
        #     "position": [0.7, 0, 0],
        #     "orientation":  [0, 0, 1.5708]
        # },
        {
            "object_name": "cube0",
            "model_file_or_sim_id": "cube_small.urdf",
            "position": [0.75, 0, .55]
        },
        {
            "object_name": "cube1",
            "model_file_or_sim_id": "cube_small.urdf",
            "position": [0.74, 0.05, .55]
        },
        {
            "object_name": "cube2",
            "model_file_or_sim_id": "cube_small.urdf",
            "position": [0.67, -0.1, .55]
        },
        {
            "object_name": "cube3",
            "model_file_or_sim_id": "cube_small.urdf",
            "position": [0.69, 0.1, .55]
        }
    ]
    sim_context = SawyerSimContext(configuration)
    sim = sim_context.get_sim_instance()
    _ = sim_context.get_logger()
    scs = sim_context.get_state_space()
    svc = sim_context.get_state_validity()
    sawyer_robot = sim_context.get_robot()
    _ = sawyer_robot.get_simulator_id()
    _ = sim_context.get_sim_objects(['Ground'])[0]


    n_samples = 10
    valid_samples = []
    starttime = timeit.default_timer()

    # Utilizes RPU convention
    T0_w = xyzrpy2trans([.7, 0, 0, 0, 0, 0], degrees=False)

    # Utilizes RPY convention
    Tw_e = xyzrpy2trans([-.2, 0, 1.0, 3.14, 0, 3.14], degrees=False)
    
    # Utilizes RPY convention
    Bw = bounds_matrix([(0, 100), (-.025, .025), (-.025, .025)],  # allow some tolerance in the z and y and only positve in x
                       [(-.01, .01), (-.01, .01), (-np.pi, np.pi)])  # any rotation about z, with limited rotation about x, and y.
    tsr = TSR(T0_w=T0_w, Tw_e=Tw_e, Bw=Bw,
              manipindex=0, bodyandlink=16)

    # Disabled collisions during planning with certain eclusions in place.
    with DisabledCollisionsContext(sim, [], []):
        print("Sampling start time is :", starttime)
        while len(valid_samples) < n_samples:
            sample = scs.sample()
            if svc.validate(sample):
                q_constrained = project_config(sawyer_robot, np.array(
                    sample), np.array(sample), tsr, .01, .01)
                normalized_q_constrained = []
                if q_constrained is not None:
                    for value in q_constrained:
                        normalized_q_constrained.append(
                            wrap_to_interval(value))
                else:
                    continue
                if svc.validate(normalized_q_constrained):
                    print(normalized_q_constrained)
                    valid_samples.append(normalized_q_constrained)
        print("The time difference is :", timeit.default_timer() - starttime)
        print("{} valid of {}".format(len(valid_samples), n_samples))
    for sample in valid_samples:
        world_pose, local_pose = sawyer_robot.solve_forward_kinematics(sample)
        trans, quat = world_pose[0], world_pose[1]
        print(trans, quat2rpy(quat))
        sawyer_robot.move_to_joint_pos(list(sample))
        while sawyer_robot.check_if_at_position(list(sample), 0.5) is False:
            time.sleep(0.1)
            pass
        time.sleep(1.5)

    # # Loop until someone shuts us down
    try:
        while True:
            pass
            # sim.step()
    except KeyboardInterrupt:
        p.disconnect()
        sys.exit(0)


if __name__ == "__main__":
    main()
