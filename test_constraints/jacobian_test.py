import os
import sys
from functools import partial

import pybullet as p
if os.environ.get('ROS_DISTRO'):
    import rospy
import numpy as np

from cairo_simulator.core.context import SawyerSimContext
from cairo_simulator.core.link import get_joint_info_by_name
from cairo_simulator.core.utils import JointInfo

from cairo_planning.collisions import DisabledCollisionsContext
from cairo_planning.local.interpolation import parametric_lerp
from cairo_planning.trajectory.curve import JointTrajectoryCurve
from cairo_planning.planners import PRM


def getMotorJointStates(robot):
    joint_states = p.getJointStates(robot, range(p.getNumJoints(robot)))
    joint_infos = [JointInfo(*p.getJointInfo(robot, i)) for i in range(p.getNumJoints(robot))]
    nonfixed_joint_info = [ji for ji in joint_infos if ji.type != p.JOINT_FIXED]
    nonfixed_joint_states = [joint_states[ji.idx] for ji in nonfixed_joint_info]
    joint_names = [ji.name for ji in nonfixed_joint_info]
    joint_positions = [state[0] for state in nonfixed_joint_states]
    joint_velocities = [state[1] for state in nonfixed_joint_states]
    joint_torques = [state[3] for state in nonfixed_joint_states]
    return joint_names, joint_positions, joint_velocities, joint_torques

def main():
    sim_context = SawyerSimContext(setup=False)
    sim_context.setup(sim_overrides={"use_gui": False})
    sim = sim_context.get_sim_instance()
    logger = sim_context.get_logger()
    state_space = sim_context.get_state_space()
    svc = sim_context.get_state_validity()
    sawyer_robot = sim_context.get_robot()
    sawyer_id = sawyer_robot.get_simulator_id()
    ground_plane = sim_context.get_sim_objects(['Ground'])[0]
    
    names, mpos, mvel, mtorq = getMotorJointStates(sawyer_id)
    print(names)
    # link_idx = get_joint_info_by_name(sawyer_id, 'right_gripper_tip').idx

    sawyerEELink = p.getNumJoints(sawyer_id) - 1
    result = p.getLinkState(sawyer_id,
                        sawyerEELink,
                        computeLinkVelocity=1,
                        computeForwardKinematics=1)
    link_trn, link_rot, com_trn, com_rot, frame_pos, frame_rot, link_vt, link_vr = result
    # Get the Jacobians for the CoM of the end-effector link.
    # Note that in this example com_rot = identity, and we would need to use com_rot.T * com_trn.
    # The localPosition is always defined in terms of the link frame coordinates.

    zero_vec = [0.0] * len(mpos)
    print(zero_vec)
    jac_t, jac_r = p.calculateJacobian(sawyer_id, sawyerEELink, com_trn, mpos, zero_vec, zero_vec)
    print(np.array([np.array(jac_r), np.array(jac_t)]))
    
    # print(p.getNumJoints(sawyer_id))
    # print(p.calculateJacobian(sawyer_id, 
    #               link_idx, 
    #               [0, 0, 0],
    #               [0] * 27,
    #               [0] * 27,
    #               [0] * 27))
    
# [1.5262755737449423, -0.1698540226273928, 2.7788151824762055, 2.4546623466066135, 0.7146948867821279, 2.7671787952787184, 2.606128412644311]


    


if __name__ == "__main__":
    main()
