import pybullet as p



from cairo_simulator.Simulator import Simulator, SimObject, ASSETS_PATH
from cairo_simulator.Manipulators import Sawyer
from cairo_gym.env import Env
import rospy

import random

#TODO gross stuff do not like :(
from cairo_gym.env import Env
import numpy as np

from math import sqrt

class SawyerVelReach(Env):

    def __init__(self, goal_point=(0.75, 0, .55)):
        
        rospy.init_node("CAIRO_Sawyer_Simulator")
        self.sim = Simulator(use_real_time=False)
        self.sawyer_robot = Sawyer("sawyer0", 0, 0, 0.8)
        self.sim_obj = SimObject('cube0', ASSETS_PATH + 'sphere_viz.urdf', goal_point)
        self.goal_point = goal_point
        self.action_space = 7
        self.observation_space = len(self._get_state())

        print(f"action space: {self.action_space}  observation space: {self.observation_space}")
        
    def step(self, action):
        self.sawyer_robot.move_with_joint_vel(action)
        self.sim.step()
        observation = self._get_state()
        reward, done = self._get_reward(observation)
        self.sim_obj.move_to_pose(position_vec=self.goal_point)
        
        return observation, reward, done


    def reset(self):
        #TODO works but stop hardcoding the joint numbers
        #TODO randomize sim obj position
        for i in range(7):
            p.resetJointState(self.sawyer_robot._simulator_id, i, 0, 0)


    def render(self, mode='human'):
        raise NotImplementedError


    def close(self):
        p.disconnect()


    def seed(self, seed=None):
        random.seed(seed)


    def _get_state(self):
        ee_pos, ee_quat = self.sawyer_robot.get_joint_pose_in_world_frame()
        joint_state = self.sawyer_robot.get_current_joint_states()
        joint_state = joint_state[:7]
        return  np.asarray(ee_pos + self.goal_point + tuple(joint_state))

    def _euclid_dist(self, x, y):
        distance = sqrt(sum([(a - b) ** 2 for a, b in zip(x, y)]))
        return distance

    def _get_reward(self, state):
    
        dist = self._euclid_dist(state[:3], state[3:6])

        if dist <= 0.1: done = 1
        else: done = 0

        return -dist, done