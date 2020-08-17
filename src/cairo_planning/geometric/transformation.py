import numpy as np
from scipy.spatial.transform import Rotation as R


def pseudoinverse(M):
    """
    Moore–Penrose pseudoinverse assuming full row rank. Generally used for Jacobian matrices of robot configurations.
    """
    return np.dot(M.T, np.linalg.inv(np.dot(M, M.T)))


def rpy_jacobian(J_r, euler_angles):
    roll = euler_angles[0]
    pitch = euler_angles[1]
    B = np.eye(3)
    B[1, 1] = np.cos(roll)
    B[1, 2] = -np.cos(pitch) * np.sin(roll)
    B[2, 1] = np.sin(roll)
    B[2, 2] = np.cos(pitch) * np.cos(roll)
    return np.dot(B, J_r)


def quat2euler(wxyz, degrees=False):
    r = R.from_quat([wxyz[1], wxyz[2], wxyz[3], wxyz[0]])
    return r.as_euler("ZYX", degrees=degrees)


def euler2quat(rpy, degrees=False):
    r = R.from_euler(
        'ZYX', rpy, degrees=degrees)
    return r.as_quat()


def pose2trans(xyzwxyz):
    trans = xyzwxyz[0:3]
    quat = xyzwxyz[3:7]
    rot_mat = R.from_quat(np.array([quat[1], quat[2], quat[3], quat[0]])).as_matrix()
    return np.vstack([np.hstack([rot_mat, np.array(trans).reshape(3, 1)]),
               np.array([0, 0, 0, 1])])


def xyzrpy2trans(xyzrpy, degrees=False):
    """
    Generates a SE(3) or 4x4 isometric transformation matrix:

    [R T]
    [0 1]

    Where R is a 3x3 rotation matrix and T represents a 3x1 translation.

    Parameters
    ----------
    translations : array-like
        x, y, z, r (roll), p (pitch), y (yaw) translations and rotations as a single vector
    degrees : bool
        Whether or not euler_angles are in degrees (True), or radians (False)
    Returns
    -------
    : ndarray
        4x4 transformation matrix
    """
    trans = xyzrpy[0:3]
    rpy = xyzrpy[3:6]
    rot_mat = R.from_euler(
        'ZYX', rpy, degrees=degrees).as_matrix()
    return np.vstack([np.hstack([rot_mat, np.array(trans).reshape(3, 1)]), np.array([0, 0, 0, 1])])


def bounds_matrix(translation_limits, rotation_limits):
    x_limits = np.array(translation_limits[0])
    y_limits = np.array(translation_limits[1])
    z_limits = np.array(translation_limits[2])
    roll_limits = np.array(rotation_limits[0])
    pitch_limits = np.array(rotation_limits[1])
    yaw_limits = np.array(rotation_limits[2])
    return np.vstack([x_limits, y_limits, z_limits, roll_limits, pitch_limits, yaw_limits])
