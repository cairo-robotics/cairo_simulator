import numpy as np

from cairo_planning.geometric.transformation import pose2trans, pseudoinverse, rpy_jacobian, quat2euler


def project_config(manipulator, q_old, q_s, T0_c, B, epsilon, q_step):
    while True:
        world_pose, local_pose = manipulator.solve_forward_kinematics(q_s)
        trans, quat = world_pose[0], world_pose[1]
        T0_sample = pose2trans(np.hstack([trans + quat]))
        d_vector = displacement_from_TSR(T0_sample, T0_c, B)
        # print(d_vector)
        # print(np.linalg.norm(d_vector))
        if np.linalg.norm(d_vector) < epsilon:
            # print(d_vector)
            # print(np.linalg.norm(d_vector))
            return q_s
        J = manipulator.get_jacobian(q_s)
        J_t = J[0:3, :]
        J_rpy = rpy_jacobian(J[3:6, :], quat2euler(quat))
        Ja = np.vstack([np.array(J_t), np.array(J_rpy)])
        J_cross = pseudoinverse(Ja)
        # J_cross = pseudoinverse(J)
        q_error = np.dot(J_cross, d_vector)
        q_s = q_s - q_error
        # if np.linalg.norm(q_s - q_old) > 2 * q_step or not within_joint_limits(manipulator, q_s):
        #     return 
        # within_joint_limits(manipulator, q_s)


def within_joint_limits(manipulator, q_s, indices):
    """
    TODO: Needs to address the index differences of the motor joints vs head pan etc,.
    """
    print("Entered this within_joint_limits function")
    for idx, limits in enumerate(manipulator._arm_joint_limits):
        if q_s[idx] < limits[0] or q_s[idx] > limits[1]:
            return False
    return True


def displacement_from_TSR(T0_obj, T0_c, B):
    Tc_obj = np.dot(np.linalg.inv(T0_c), T0_obj)
    disp = displacement(Tc_obj)
    return delta_x(disp, B)


def displacement(T):
    tc = T[0:3, 3]
    roll = np.arctan2(T[2, 1], T[2, 2])
    pitch = -np.arcsin(T[2, 0])
    yaw = np.arctan2(T[1, 0], T[0, 0])
    return np.hstack([tc, roll, pitch, yaw])


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
