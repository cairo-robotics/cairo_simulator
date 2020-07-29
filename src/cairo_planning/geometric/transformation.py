import numpy as np
from scipy.spatial.transform import Rotation as R


def transform_mat(euler_angles, translations, degrees=False):
    """
    Generations a SE(3) or 4x4 transformation matrix:

    [R T]
    [0 1]

    Where R is a 3x3 rotation matrix and T represents a 3x1 translation.

    Parameters
    ----------
    euler_angles : array-like
        x (roll), y (pitch), z (yaw) rotations
    translations : array-like
        x, y, z translations
    degrees : bool
        Whether or not euler_angles are in degrees (True), or radians (False)
    Returns
    -------
    : ndarray
        4x4 transformation matrix
    """
    rot_mat = R.from_euler(
        'zyx', euler_angles[::-1], degrees=degrees).as_matrix()
    return np.vstack([np.hstack([rot_mat, np.array(euler_angles).reshape(3, 1)]), np.array([0, 0, 0, 1])])

