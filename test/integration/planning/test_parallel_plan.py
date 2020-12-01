import os
import sys
from functools import partial
import time

if os.environ.get('ROS_DISTRO'):
    import rospy
import numpy as np

from cairo_simulator.core.context import SawyerSimContext

from cairo_planning.planners.roadmap import PRMParallel
from cairo_planning.local.interpolation import parametric_lerp
from cairo_planning.local.curve import JointTrajectoryCurve


if __name__ == "__main__":
    sim_context = SawyerSimContext(None, setup=False)
    sim_context.setup(sim_overrides={"run_parallel": True})
    sim = sim_context.get_sim_instance()
    logger = sim_context.get_logger()
    sawyer_robot = sim_context.get_robot()
    svc = sim_context.get_state_validity()

    interp_fn = partial(parametric_lerp, steps=5)
    prm = PRMParallel(SawyerSimContext, None, svc, interp_fn, params={
                       'n_samples': 8000, 'k': 8, 'ball_radius': 1.5})
    plan = prm.plan(np.array([0, 0, 0, 0, 0, 0, 0]), np.array([1.5262755737449423, -0.1698540226273928,
                                                               2.7788151824762055, 2.4546623466066135, 0.7146948867821279, 2.7671787952787184, 2.606128412644311]))
    path = prm.get_path(plan)
    
    sawyer_robot.move_to_joint_pos([0, 0, 0, 0, 0, 0, 0])
    time.sleep(3)
    while sawyer_robot.check_if_at_position([0, 0, 0, 0, 0, 0, 0], 0.5) is False:
        time.sleep(0.1)
        sim.step()
    time.sleep(3)

    # splinging uses numpy so needs to be converted
    path = [np.array(p) for p in path]
    logger.info("Length of path: {}".format(len(path)))
    if len(path) > 0:
        # Create a MinJerk spline trajectory using JointTrajectoryCurve and execute
        jtc = JointTrajectoryCurve()
        traj = jtc.generate_trajectory(path, move_time=10)
        sawyer_robot.execute_trajectory(traj)
        try:
            while True:
                sim.step()
        except KeyboardInterrupt:
            sys.exit(0)
    else:
        logger.err("No path found.")
