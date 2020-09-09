import os

import pybullet as p
if os.environ.get('ROS_DISTRO'):
    import rospy
import numpy as np

from cairo_simulator.core.context import SawyerSimContext
from cairo_simulator.core.link import get_joint_info_by_name
from cairo_simulator.core.utils import JointInfo


def getMotorJointStates(robot):
    joint_states = p.getJointStates(robot, range(p.getNumJoints(robot)))
    joint_infos = [JointInfo(*p.getJointInfo(robot, i))
                   for i in range(p.getNumJoints(robot))]
    nonfixed_joint_info = [
        ji for ji in joint_infos if ji.type != p.JOINT_FIXED]
    nonfixed_joint_states = [joint_states[ji.idx]
                             for ji in nonfixed_joint_info]
    joint_names = [ji.name for ji in nonfixed_joint_info]
    joint_positions = [state[0] for state in nonfixed_joint_states]
    joint_velocities = [state[1] for state in nonfixed_joint_states]
    joint_torques = [state[3] for state in nonfixed_joint_states]
    return joint_names, joint_positions, joint_velocities, joint_torques


def getJointStates(robot):
    joint_states = p.getJointStates(robot, range(p.getNumJoints(robot)))
    joint_positions = [state[0] for state in joint_states]
    joint_velocities = [state[1] for state in joint_states]
    joint_torques = [state[3] for state in joint_states]
    return joint_positions, joint_velocities, joint_torques


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

    # Get the robot moving!
    sawyer_robot.move_to_joint_pos(target_position=[1]*7)

    while sawyer_robot.check_if_at_position([1]*7, 0.5) is False:
        p.stepSimulation()
        print('\n\n\n\n\n\n')
        # Get current states
        pos, vel, torq = getJointStates(sawyer_id)
        names, mpos, mvel, mtorq = getMotorJointStates(sawyer_id)
        print("Moving joint names:")
        print(names)

        sawyerEELink = get_joint_info_by_name(sawyer_id, 'right_l6').idx
        print(sawyerEELink)
        result = p.getLinkState(sawyer_id,
                                sawyerEELink,
                                computeLinkVelocity=1,
                                computeForwardKinematics=1)
        link_trn, link_rot, com_trn, com_rot, frame_pos, frame_rot, link_vt, link_vr = result
        # Get the Jacobians for the CoM of the end-effector link.
        # Note that in this example com_rot = identity, and we would need to use com_rot.T * com_trn.
        # The localPosition is always defined in terms of the link frame coordinates.
        zero_vec = [0.0] * len(mpos)

        jac_t, jac_r = p.calculateJacobian(
            sawyer_id, sawyerEELink, com_trn, mpos, zero_vec, zero_vec)
        J = np.vstack([np.array(jac_t), np.array(jac_r)])[
            :, [0, 2, 3, 4, 5, 6, 7]]
        print("JACOBIAN")
        print(J)
        J_cross = np.dot(J.T, np.linalg.inv(np.dot(J, J.T)))
        print("PSEUDOINVERSE JACOBIAN")
        print(J_cross)

        print("Link linear position and velocity of CoM from getLinkState:")
        print(link_vt, link_vr)
        print("Link linear and angular velocity of CoM from linearJacobian * q_dot:")
        print(np.dot(J, np.array(mvel)[[0, 2, 3, 4, 5, 6, 7]].T))
        print()
        print()
        print("Link joint velocities of CoM from getLinkState:")
        print(np.array(mvel)[[0, 2, 3, 4, 5, 6, 7]])
        print("Link q given link linear position and velocity using J_cross * x_dot.")
        x = np.hstack([np.array(link_vt), np.array(link_vr)])
        q = np.dot(J_cross, np.array(x).T)
        print(q)


if __name__ == "__main__":
    main()
