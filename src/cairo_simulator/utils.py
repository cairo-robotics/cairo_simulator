import os
from collections import namedtuple

import numpy as np
from pyquaternion import Quaternion

ASSETS_PATH = os.path.dirname(os.path.abspath(__file__)) + '/../../assets/' # Find ./cairo_simulator/assets/ from ./cairo_simulator/src/cairo_simulator/


JointInfo = namedtuple('JointInfo', ['idx', 'name', 'type',
                                     'qidx', 'uidx', 'flags',
                                     'damping', 'friction', 'lower_limit', 'upper_limit',
                                     'max_force', 'max_velocity', 'link_name', 'joint_axis',
                                     'parent_position', 'parent_orientation', 'parent_idx'])

def compute_3d_homogeneous_transform(x, y, z, alpha, beta, gamma):
    '''
    http://planning.cs.uiuc.edu/node104.html
    '''

    T = np.zeros([4,4])
    T[0,0] = np.cos(alpha)*np.cos(beta)
    T[1,0] = np.sin(alpha)*np.cos(beta)
    T[2,0] = -1*np.sin(beta)

    T[0,1] = np.cos(alpha) * np.sin(beta) * np.sin(gamma) - np.sin(alpha) * np.cos(gamma)
    T[1,1] = np.sin(alpha) * np.sin(beta) * np.sin(gamma) + np.cos(alpha) * np.cos(gamma)
    T[2,1] = np.cos(beta) * np.sin(gamma)

    T[0,2] = np.cos(alpha) * np.sin(beta) * np.cos(gamma) + np.sin(alpha) * np.sin(gamma)
    T[1,2] = np.sin(alpha) * np.sin(beta) * np.cos(gamma) - np.cos(alpha) * np.sin(gamma)
    T[2,2] = np.cos(beta) * np.cos(gamma)

    T[0,3] = x
    T[1,3] = y
    T[2,3] = z
    T[3,3] = 1

    return T

def invert_3d_homogeneous_transform(T):
    '''
    http://vr.cs.uiuc.edu/node81.html
    Inverts a 3D homogeneous transform
    T: 4x4 numpy Matrix
    '''

    R = T[0:3,0:3]
    rot_inv = np.zeros([4,4])
    rot_inv[0:3, 0:3] = R.T
    rot_inv[3,3] = 1

    trans_inv = np.eye(4)
    for i in range(3):
        trans_inv[i,3] = -T[i,3]
    
    return np.matmul(rot_inv, trans_inv)


def quaternion_from_matrix(matrix):

    quat = Quaternion(matrix=matrix)
    return [quat[1], quat[2], quat[3], quat[0]]
