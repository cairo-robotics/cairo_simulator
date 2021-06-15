import os
from functools import partial
import time

import pybullet as p
if os.environ.get('ROS_DISTRO'):
    import rospy
import numpy as np

from cairo_simulator.core.sim_context import SawyerSimContext

from cairo_planning.collisions import DisabledCollisionsContext
from cairo_planning.local.interpolation import parametric_lerp
from cairo_planning.planners import LazyPRM
from cairo_planning.core.serialization import dump_PRM


def main():
    sim_context = SawyerSimContext()
    sim = sim_context.get_sim_instance()
    logger = sim_context.get_logger()
    state_space = sim_context.get_state_space()
    svc = sim_context.get_state_validity()
    sawyer_robot = sim_context.get_robot()
    _ = sawyer_robot.get_simulator_id()
    _ = sim_context.get_sim_objects(['Ground'])[0]

    sawyer_robot.move_to_joint_pos([0, 0, 0, 0, 0, 0, 0])
    time.sleep(1)
 
    with DisabledCollisionsContext(sim, [], []):
        #######
        # LazyPRM #
        #######
        # Use parametric linear interpolation with 10 steps between points.
        interp = partial(parametric_lerp, steps=10)
        # See params for PRM specific parameters
        prm = LazyPRM(state_space, svc, interp, params={
                  'n_samples': 4000, 'k': 8, 'ball_radius': 2.5})
        logger.info("Planning....")
        plan = prm.plan(np.array([0, 0, 0, 0, 0, 0, 0]), np.array([1.5262755737449423, -0.1698540226273928,
                                                                   2.7788151824762055, 2.4546623466066135, 0.7146948867821279, 2.7671787952787184, 2.606128412644311]))

    # Dump thje samples and configuration
    dump_PRM(sim_context.config, prm, os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":
    main()
