import time
import os
import sys
from functools import partial
import numpy as np

if os.environ.get('ROS_DISTRO'):
    import rospy
import pybullet as p

from cairo_simulator.core.utils import ASSETS_PATH
from cairo_simulator.core.log import Logger
from cairo_simulator.core.simulator import Simulator, SimObject
from cairo_simulator.devices.manipulators import Sawyer

from cairo_simulator.core.sim_context import SawyerSimContext
from cairo_planning.collisions import DisabledCollisionsContext
from cairo_planning.local.interpolation import parametric_lerp
from cairo_planning.planners import PRM


class Task:

    def __init__(self) -> None:
        base_config = {}
        base_config["sim"] = {
            "use_real_time": False
        }
        # This configuration will exclude the two links causing invalid state validy checks
        # for your given startup point in this script...
        base_config["state_validity"] = {
            "self_collision_exclusions": [("head", "right_l2")]
        }
        self.sim_context = SawyerSimContext(configuration=base_config)
        self.sim = self.sim_context.get_sim_instance()
        self.state_space = self.sim_context.get_state_space()
        self.svc = self.sim_context.get_state_validity()
        self.sawyer_robot = self.sim_context.get_robot()

        self.default_joint_positions = [0.002, -1.182, 0.002, 2.177, 0.001, 0.567, 3.316]
        self._setup_env()

    def move_robot_to_neutral(self):
        self.sawyer_robot.move_to_joint_pos(self.default_joint_positions)

    def _setup_env(self):
        SimObject("Ground", "plane.urdf", [0,0,0])
        SimObject("Table", ASSETS_PATH + 'table.sdf', (0.7, 0, 0), (0, 0, 1.5708)) # Table rotated 90deg along z-axis

        sim_obj0 = SimObject('cube0', 'cube_small.urdf', (0.7, 0.4, .55))
        sim_obj1 = SimObject('cube1', 'cube_small.urdf', (0.7, 0.2, .55))
        sim_obj2 = SimObject('cube2', 'cube_small.urdf', (0.7, 0, .55))
        sim_obj3 = SimObject('cube3', 'cube_small.urdf', (0.7, -0.2, .55))
        sim_obj4 = SimObject('cube4', 'cube_small.urdf', (0.7, -0.4, .55))
        self.sim_objects = [sim_obj0, sim_obj1, sim_obj2, sim_obj3, sim_obj4]

    def generate_prm(self):

        with DisabledCollisionsContext(self.sim, [], []):
            #######
            # PRM #
            #######
            # Use parametric linear interpolation with 10 steps between points.
            interp = partial(parametric_lerp, steps=10)
            # See params for PRM specific parameters
            prm = PRM(self.state_space, self.svc, interp, params={
                    'n_samples': 6000, 'k': 20, 'ball_radius': 1.75})
            prm.plan(q_start=self.default_joint_positions)

            for cube in self.sim_objects:
                cube_pos, _ = cube.get_pose()
                # goal position is above the cube
                goal_pos = [cube_pos[0], cube_pos[1], cube_pos[2]+0.15]
                q_goal = self.sawyer_robot.solve_inverse_kinematics(goal_pos)
                prm.attach_end(cube._name, q_goal)

            prm.graph.write_pickle("sawyer_prm.pickle")

    def load_prm(self):
        with DisabledCollisionsContext(self.sim, [], []):
            #######
            # PRM #
            #######
            # Use parametric linear interpolation with 10 steps between points.
            interp = partial(parametric_lerp, steps=10)
            # See params for PRM specific parameters
            self.prm = PRM(self.state_space, self.svc, interp, params={
                    'n_samples': 6000, 'k': 10, 'ball_radius': 1.75})
            self.prm.graph = self.prm.graph.Read_Pickle("sawyer_prm1.pickle")

def main():
    task = Task()
    task.generate_prm()

    ee_pos = task.sawyer_robot.solve_forward_kinematics(task.default_joint_positions)[0][0]
    p.addUserDebugPoints([ee_pos], [[1, 0, 0]], pointSize=3)

    # task.prm.init_nearest_neighbor()
    # task.prm.attach_start("start", task.default_joint_positions)

    # print(task.prm._neighbors(task.default_joint_positions, k_override=30, within_ball=True)[:5])

    # dists = []
    # for i in range(len(task.prm.graph.vs)):
    #     if task.prm.graph.vs[i]["name"] != "start":
    #         dists.append((task.prm.graph.vs[i]["value"], np.linalg.norm(np.array(task.default_joint_positions)-np.array(task.prm.graph.vs[i]["value"]))))
    # print(sorted(dists, key=lambda x: x[1])[:3])      

    # ee_positions = []
    # for q_near in task.prm._neighbors(task.default_joint_positions, k_override=30, within_ball=True):
    #     ee_pos = task.sawyer_robot.solve_forward_kinematics(q_near)[0][0]
    #     ee_positions.append(ee_pos)
    # p.addUserDebugPoints(ee_positions, [[1, 0, 0]]*len(ee_positions), pointSize=3)

    # Loop until someone shuts us down
    try:
        while True:
            task.sim.step()
    except KeyboardInterrupt:
        p.disconnect()
        sys.exit(0)
   



if __name__ == "__main__":
    main()
