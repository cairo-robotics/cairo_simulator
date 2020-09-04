import numpy as np
from scipy.spatial.transform import Rotation as R


def pseudoinverse(M):
    """
    Moore–Penrose pseudoinverse assuming full row rank. Generally used for Jacobian matrices of robot configurations.
    """
    return np.linalg.pinv(M)


def analytic_xyz_jacobian(J_r, rpy):
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
    E[0, 0] = 1 
    E[0, 2] = np.sin(rpy[1])
    E[1, 1] = -np.cos(rpy[0])
    E[1, 2] = -np.cos(rpy[1]) * np.sin(rpy[0])
    E[2, 1] = np.sin(rpy[0]) 
    E[2, 2] = np.cos(rpy[0]) * np.cos(rpy[1])
    return np.dot(np.linalg.inv(E), J_r)


def quat2rpy(wxyz, degrees=False):
    r = R.from_quat([wxyz[1], wxyz[2], wxyz[3], wxyz[0]])
    return r.as_euler("xyz", degrees=degrees)


def rpy2quat(rpy, degrees=False):
    r = R.from_euler(
        'ZYX', ypr, degrees=degrees)
    return r.as_quat()


    quat = R.from_euler(
        'xyz', rpy, degrees=degrees).as_quat()
    return np.array((quat[3], quat[0], quat[1], quat[2]))
def pose2trans(xyzwxyz):
    trans = xyzwxyz[0:3]
    quat = xyzwxyz[3:7]
    rot_mat = R.from_quat(
        np.array((quat[1], quat[2], quat[3], quat[0]))).as_matrix()
    return np.vstack([np.hstack([rot_mat, np.array(trans).reshape(3, 1)]),
                      np.array((0, 0, 0, 1))])

def rot2rpy(Rt, degrees=False):
    """
    Converts a rotation matrix into rpy / intrinsic xyz euler angle form.

    Args:
        Rt (ndarray): The rotation matrix.

    Returns:
        ndarray: Angles in rpy / intrinsic xyz euler angle form.
    """
    return R.from_matrix(Rt).as_euler('xyz', degrees=degrees)

def rpy2rot(rpy, degrees=False):
    """
    Converts rpy / intrinsic xyz euler angles into a rotation matrix.

    Args:
        rpy (array-like): rpy / intrinsic xyz euler angles.
        degrees (bool) : Whether or not euler_angles are in degrees (True), or radians (False).

    Returns:
        ndarray: The rotation matrix.
    """
    return R.from_euler('xyz', rpy, degrees=degrees).as_matrix()

def xyzrpy2trans(xyzrpy, degrees=False):
    """
    Generates a SE(3) or 4x4 isometric transformation matrix:

    [R T]
    [0 1]

    Where R is a 3x3 rotation matrix and T represents a 3x1 translation.

    Args:
        xyzrpy (array-like): x, y, z translations and r (roll), p (pitch), y (yaw) rotations as a single vector.
        degrees (bool) : Whether or not euler_angles are in degrees (True), or radians (False).
    Returns:
        ndarray: 4x4 transformation matrix.
    """
    trans = xyzrpy[0:3]
    rpy = xyzrpy[3:6]
    rot_mat = rpy2rot(rpy, degrees=degrees)
    return np.vstack([np.hstack([rot_mat, np.array(trans).reshape(3, 1)]), np.array((0, 0, 0, 1))])


def bounds_matrix(translation_limits, rotation_limits):
    x_limits = np.array(translation_limits[0])
    y_limits = np.array(translation_limits[1])
    z_limits = np.array(translation_limits[2])
    roll_limits = np.array(rotation_limits[0])
    pitch_limits = np.array(rotation_limits[1])
    yaw_limits = np.array(rotation_limits[2])
    return np.vstack([x_limits, y_limits, z_limits, roll_limits, pitch_limits, yaw_limits])
