from itertools import product

import numpy as np

from cairo_planning.geometric.transformation import pose2trans, pseudoinverse, analytic_zyx_jacobian, quat2ypr


def project_config(manipulator, q_old, q_s, TSR, epsilon, q_step):
    count = 0
    while True:
        count += 1
        world_pose, _ = manipulator.solve_forward_kinematics(q_s)
        trans, quat = world_pose[0], world_pose[1]
        T0_s = pose2trans(np.hstack([trans + quat]))
        min_distance, x_err = distance_from_TSR(T0_s, tsr)
        # print(x_err)
        # print(min_distance)
        if min_distance < epsilon:
            return q_s
        elif count > iter_count:
            return None
        J = manipulator.get_jacobian(q_s)
        J_t = J[0:3, :]
        # obtain the analytic jacobian
        J_rpy = analytic_xyz_jacobian(J[3:6, :], quat2rpy(quat))
        Ja = np.vstack([np.array(J_t), np.array(J_rpy)])
        J_cross = pseudoinverse(Ja)
        q_error = np.dot(J_cross, x_err)
        q_s = q_s - e_step * q_error
        # if np.linalg.norm(q_s - q_old) > 2 * q_step or not within_joint_limits(manipulator, q_s):
        #     return
        if not within_joint_limits(manipulator, q_s):
            return None


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
    # pose of the grasp location or the pose of the object held by the hand in world coordinates
    T0_sp = np.dot(T0_s, np.linalg.inv(tsr.Tw_e))
    # T0_sp in terms of the coordinates of the target frame w given by the Task Space Region tsr.
    Tw_sp = np.dot(np.linalg.inv(tsr.T0_w), T0_sp)
    # Generate the displacement vector of Tw_sp. Displacement represents the error given T0_s relative to Tw_e transform.
    disp = displacement(Tw_sp)
    # Since there are equivalent angle displacements for rpy, generate those equivalents by added +/- PI.
    # Use the smallest delta_x_dist of the equivalency set.
    rpys = generate_equivalent_euler_angles([disp[3], disp[4], disp[5]])
    deltas = []
    deltas.append(delta_x(disp, tsr.Bw))
    for rpy in rpys:
        deltas.append(
            delta_x(np.hstack([disp[0:3], rpy[0], rpy[1], rpy[2]]), tsr.Bw))
    distances = [delta_x_dist(delta) for delta in deltas]
    return min(distances), deltas[distances.index(min(distances))]


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
