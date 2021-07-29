#! /usr/bin/env python
import os
import json

import roslib
import rospy
import actionlib
import intera_interface

from control_msgs.msg import (
    FollowJointTrajectoryAction,
    FollowJointTrajectoryGoal
)


def execute_trajectory(trajectory):
    rospy.init_node('trajectory_test')
    namespace = 'robot/limb/right/follow_joint_trajectory'
    client = actionlib.SimpleActionClient(namespace, FollowJointTrajectoryAction)
    client.wait_for_server()

    limb = intera_interface.Limb('right', synchronous_pub=True)


    goal = FollowJointTrajectoryGoal()
    goal.trajectory.joint_names = limb.joint_names()
    goal.trajectory.points = trajectory

    # Fill in the goal here
    client.send_goal(goal)
    client.wait_for_result(rospy.Duration.from_sec(5.0))

if __name__ == "__main__":
    abs_file_path = os.path.join(os.path.dirname(__file__), "traj.json")
    with open(abs_file_path, 'r') as f:
        data = json.load(f)
    trajectory = [point['point'] for point in data['trajectory']]
    
    execute_trajectory(trajectory)
