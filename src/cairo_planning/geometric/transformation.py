import numpy as np
from scipy.spatial.transform import Rotation as R


def pseudoinverse(M):
    """
    Moore–Penrose pseudoinverse assuming full row rank. Generally used for Jacobian matrices of robot configurations.
    """
    return np.linalg.pinv(M)


def analytic_zyx_jacobian(J_r, ypr):
    """Generates the Euler mapping matric from J(q) -> J(x) for the equation
    to produce the analytical jacobian given ZYX extrinsic euler angles x (ypr).
    
    Ja(q) = E(x)*J(q)

    Provides the pseudoinverse mapping:
    q' = Ja(q)^+ * x'

    Args:
        J_r ([type]): [description]
        ypr ([type]): [description]

    Returns:
        [type]: [description]
    """
    yaw = ypr[0]
    pitch = ypr[1]
    E = np.zeros([3, 3])
    E[0, 2] = 1
    E[1, 1] = -np.sin(yaw)
    E[1, 2] = np.cos(yaw)
    E[2, 0] = np.cos(pitch) * np.cos(yaw)
    E[2, 1] = np.cos(pitch) * np.sin(yaw)
    E[2, 2] = -np.sin(yaw)
    return np.dot(np.linalg.inv(E), J_r)


def quat2ypr(wxyz, degrees=False):
    r = R.from_quat([wxyz[1], wxyz[2], wxyz[3], wxyz[0]])
    return r.as_euler("ZYX", degrees=degrees)


def ypr2quat(ypr, degrees=False):
    r = R.from_euler(
        'ZYX', ypr, degrees=degrees)
    return r.as_quat()


def pose2trans(xyzwxyz):
    trans = xyzwxyz[0:3]
    quat = xyzwxyz[3:7]
    rot_mat = R.from_quat(np.array([quat[1], quat[2], quat[3], quat[0]])).as_matrix()
    return np.vstack([np.hstack([rot_mat, np.array(trans).reshape(3, 1)]),
               np.array([0, 0, 0, 1])])


def xyzypr2trans(xyzypr, degrees=False):
    """
    Generates a SE(3) or 4x4 isometric transformation matrix:

    [R T]
    [0 1]

    Where R is a 3x3 rotation matrix and T represents a 3x1 translation.

    Parameters
    ----------
    translations : array-like
        x, y, z, y (yaw), p (pitch), r (roll),  translations and rotations as a single vector
    degrees : bool
        Whether or not euler_angles are in degrees (True), or radians (False)
    Returns
    -------
    : ndarray
        4x4 transformation matrix
    """
    trans = xyzypr[0:3]
    ypr = xyzypr[3:6]
    rot_mat = R.from_euler(
        'ZYX', ypr, degrees=degrees).as_matrix()
    return np.vstack([np.hstack([rot_mat, np.array(trans).reshape(3, 1)]), np.array([0, 0, 0, 1])])


def bounds_matrix(translation_limits, rotation_limits):
    x_limits = np.array(translation_limits[0])
    y_limits = np.array(translation_limits[1])
    z_limits = np.array(translation_limits[2])
    yaw_limits = np.array(rotation_limits[0])
    pitch_limits = np.array(rotation_limits[1])
    roll_limits = np.array(rotation_limits[2])
    return np.vstack([x_limits, y_limits, z_limits, yaw_limits, pitch_limits, roll_limits])
