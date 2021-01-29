import os
import time

import pybullet as p
if os.environ.get('ROS_DISTRO'):
    import rospy
import timeit

from cairo_simulator.core.utils import ASSETS_PATH
from cairo_simulator.core.sim_context import SawyerSimContext
from cairo_simulator.core.simulator import SimObject

from cairo_planning.collisions import DisabledCollisionsContext, get_closest_points


def main():
    config = {}
    config["sim_objects"] = [
        {
            "object_name": "Ground",
            "model_file_or_sim_id": "plane.urdf",
            "position": [0, 0, 0]
        },
        {
            "object_name": "Table",
            "model_file_or_sim_id": ASSETS_PATH + 'table.sdf',
            "position": [0.0, 0, 0],
            "orientation":  [0, 0, 1.5708]
        }
    ]
    sim_context = SawyerSimContext(config)
    sim = sim_context.get_sim_instance()
    _ = sim_context.get_logger()
    scs = sim_context.get_state_space()
    svc = sim_context.get_state_validity()
    sawyer_robot = sim_context.get_robot()
    _ = sawyer_robot.get_simulator_id()

    table_obj = sim_context.get_sim_objects(names="Table")[0]

    sawyer_robot.move_to_joint_pos([0, 0, 0, 0, 0, 0, 0])
    time.sleep(1)

    n_samples = 1000
    valid_samples = []
    starttime = timeit.default_timer()

    # Disabled collisions during planning with certain eclusions in place.
    with DisabledCollisionsContext(sim, [], []):
        print("Sampling start time is :", starttime)
        for i in range(0, n_samples):
            sample = scs.sample()
            if svc.validate(sample):
                print(sample)
                valid_samples.append(sample)
        print("The time difference is :", timeit.default_timer() - starttime)
        print("{} valid of {}".format(len(valid_samples), n_samples))

    p.disconnect()


if __name__ == "__main__":
    main()
