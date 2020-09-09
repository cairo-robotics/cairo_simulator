import sys

import numpy as np
import pybullet as p
from scipy.spatial.transform import Rotation as R

from cairo_planning.geometric.tsr import TSR
from cairo_planning.geometric.transformation import xyzrpy2trans, bounds_matrix, quat2rpy, rpy2quat, rot2rpy, rpy2rot
from cairo_planning.constraints.projection import displacement

def rot2rpy_berenson(R):
    roll = np.arctan2(R[2, 1], R[2, 2])
    pitch = -np.arcsin(R[2, 0])
    yaw = np.arctan2(R[1, 0], R[0, 0])
    return [roll, pitch, yaw]

if __name__ == "__main__":
    print(quat2rpy([np.cos(np.pi/4), 0, 0, np.sin(np.pi/4)]))
    print(p.getEulerFromQuaternion([0, 0, np.sin(np.pi/4), np.cos(np.pi/4)]))
    print()
    print(quat2rpy([-0.022076, 0.016451, 0.99929, 0.025687]))
    print(p.getEulerFromQuaternion([0.016451, 0.99929, 0.025687, -0.022076]))
    print()
    print(rpy2quat(quat2rpy([-0.022076, 0.016451, 0.99929, 0.025687])))
    print(p.getQuaternionFromEuler(p.getEulerFromQuaternion([0.016451, 0.99929, 0.025687, -0.022076])))
    print()
    ninety_x_rot = np.array([[  1.0000000,  0.0000000,  0.0000000],
                        [0.0000000, -0.4480736, -0.8939967],
                        [0.0000000,  0.8939967, -0.4480736 ]])

    print(TSR.rot_to_rpy(ninety_x_rot))
    print(rot2rpy_berenson(ninety_x_rot))
    print(rot2rpy(ninety_x_rot))
    print(rot2rpy(ninety_x_rot))
    print()
    
    print()
    ninety_y_rot = np.array([[-0.4480736,  0.0000000,  0.8939967],
                        [0.0000000,  1.0000000,  0.0000000],
                        [-0.8939967,  0.0000000, -0.4480736]])
    print(TSR.rot_to_rpy(ninety_y_rot))
    print(rot2rpy_berenson(ninety_y_rot))
    print(rot2rpy(ninety_y_rot))
    
    print()
    ninety_z_rot = np.array([[ -0.4480736, -0.8939967,  0.0000000],
                        [0.8939967, -0.4480736,  0.0000000],
                        [0.0000000,  0.0000000,  1.0000000]])
    print(TSR.rot_to_rpy(ninety_y_rot))
    print(rot2rpy_berenson(ninety_y_rot))
    print(rot2rpy(ninety_y_rot))
    
    print()
    ninety_xyz_rot = np.array([[0.2007700,  0.4005763,  0.8939967],
                        [-0.7586902, -0.5137390,  0.4005763],
                        [0.6197423, -0.7586902,  0.2007700]])
    print(TSR.rot_to_rpy(ninety_xyz_rot))
    print(rot2rpy_berenson(ninety_xyz_rot))
    print(rot2rpy(ninety_xyz_rot))
    
    
    ninety_x_euler = [90, 0, 0]
    print()
    sys.exit(0)
    