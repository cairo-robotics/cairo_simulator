from itertools import product

import numpy as np

from cairo_planning.geometric.transformation import pose2trans, pseudoinverse, analytic_zyx_jacobian, quat2ypr


def project_config(manipulator, q_old, q_s, TSR, epsilon, q_step):
    count = 0
    while True:
        count += 1
        world_pose, local_pose = manipulator.solve_forward_kinematics(q_s)
        trans, quat = world_pose[0], world_pose[1]
        T0_s = pose2trans(np.hstack([trans + quat]))
        d_vector = displacement_from_TSR(T0_s, TSR)
        print(d_vector)
        print(np.linalg.norm(d_vector))
        if np.linalg.norm(d_vector) < epsilon:
            print(d_vector)
            print(np.linalg.norm(d_vector))
            print(trans, quat2ypr(quat))
            return q_s
        elif count > 5000:
            return None
        J = manipulator.get_jacobian(q_s)
        J_t = J[0:3, :]
        # obtain the analytic jacobian
        J_rpy = analytic_zyx_jacobian(J[3:6, :], quat2ypr(quat))
        Ja = np.vstack([np.array(J_t), np.array(J_rpy)])
        J_cross = pseudoinverse(Ja)
        # J_cross = pseudoinverse(J)
        q_error = np.dot(J_cross, d_vector)
        q_s = q_s - .1 * q_error
        # if np.linalg.norm(q_s - q_old) > 2 * q_step or not within_joint_limits(manipulator, q_s):
        #     return 
        # within_joint_limits(manipulator, q_s)


def within_joint_limits(manipulator, q_s):
    """
    TODO: Fragile in that _arm_joint_limits assumes same ordering of limits as the indices of q_s
    """
    for idx, limits in enumerate(manipulator._arm_joint_limits):
        if q_s[idx] < limits[0] or q_s[idx] > limits[1]:
            return False
    return True

def displacement_from_TSR(T0_s, TSR):
    T0_sp = np.dot(T0_s, np.linalg.inv(TSR.Tw_e))
    Tw_sp = np.dot(np.linalg.inv(TSR.T0_w), T0_sp)
    disp = displacements(Tw_sp)
    # Since there are equivalent angle displacements for rpy, generate those equivalents by added +/- PI.
    # Use the smallest delta_x_dist of the equivalency set.
    yprs = generate_equivalent_displacement_angles([disp[3], disp[4], disp[5]])
    deltas = []
    deltas.append(delta_x(disp, TSR.Bw))
    for ypr in yprs:
        deltas.append(delta_x(np.hstack([disp[0:3], ypr[0], ypr[1], ypr[2]]), TSR.Bw))
    distances = [delta_x_dist(delta) for delta in deltas]
    return deltas[distances.index(min(distances))]


def displacements(T):
    Tc_obj = T[0:3, 3]
    Rc_obj = T[0:3, 0:3]
    yaw = np.arctan2(Rc_obj[1, 0], Rc_obj[0, 0])
    pitch = -np.arcsin(Rc_obj[2, 0])
    roll = np.arctan2(Rc_obj[2, 1], Rc_obj[2, 2])
    return np.hstack([Tc_obj, yaw, pitch, roll])

def generate_equivalent_displacement_angles(ypr):
    yaws = [ypr[0] + np.pi, ypr[0] - np.pi]
    pitches = [ypr[1] + np.pi, ypr[1] - np.pi]
    rolls = [ypr[2] + np.pi, ypr[2] - np.pi]
    return list(product(yaws, pitches, rolls))

def delta_x(displacement, constraint_matrix):
    delta = []
    for i in range(0, displacement.shape[0]):
        cmin = constraint_matrix[i, 0]
        cmax = constraint_matrix[i, 1]
        di = displacement[i]
        if di > cmax:
            delta.append(di - cmax)
        elif di < cmin:
            delta.append(di - cmin)
        else:
            delta.append(0)
    return np.array(delta)


def delta_x_dist(del_x):
    return np.linalg.norm(del_x)
