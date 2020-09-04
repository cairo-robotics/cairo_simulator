import unittest

import numpy as np
import pybullet as p

from cairo_planning.geometric.transformation import xyzrpy2trans, bounds_matrix, quat2rpy, rpy2quat, rot2rpy, rpy2rot
from cairo_planning.geometric.tsr import TSR


class TestQuat2RPY(unittest.TestCase):
    def test(self):
        pybullet_rpy = p.getEulerFromQuaternion([0.3922323, 0.5883484, -0.5883484, 0.3922323])
        cairo_rpy = quat2rpy([0.3922323, 0.3922323, 0.5883484, -0.5883484])
        pybullet_rpy = [format(value, '.4f') for value in pybullet_rpy]
        cairo_rpy = [format(value, '.4f') for value in cairo_rpy]
        self.assertEqual(list(pybullet_rpy), list(cairo_rpy))
        
        pybullet_rpy = p.getEulerFromQuaternion([0.1093958, 0.6997598, -0.6997598, 0.0933013])
        cairo_rpy = quat2rpy([0.0933013, 0.1093958, 0.6997598, -0.6997598])
        pybullet_rpy = [format(value, '.4f') for value in pybullet_rpy]
        cairo_rpy = [format(value, '.4f') for value in cairo_rpy]
        self.assertEqual(list(pybullet_rpy), list(cairo_rpy))
        
class TestRPY2Rot(unittest.TestCase):
    def test(self):
        rpy = [1.5740595, -0.0225264, -2.8538895]
        tsr_trans = TSR.rpy_to_rot(rpy)
        cairo_trans = rpy2rot(rpy)
        np.testing.assert_array_almost_equal(tsr_trans, cairo_trans, decimal=4)
        
class TestRot2RPY(unittest.TestCase):
    def test_90_degree_x(self):
        ninety_x_rot = np.array([[  1.0000000,  0.0000000,  0.0000000],
                        [0.0000000, -0.4480736, -0.8939967],
                        [0.0000000,  0.8939967, -0.4480736 ]])
        tsr_R = TSR.rot_to_rpy(ninety_x_rot)
        cairo_R = rot2rpy(ninety_x_rot)
        np.testing.assert_array_almost_equal(tsr_R, cairo_R, decimal=4)
    
    def test_90_degree_y(self):
        ninety_y_rot = np.array([[-0.4480736,  0.0000000,  0.8939967],
                        [0.0000000,  1.0000000,  0.0000000],
                        [-0.8939967,  0.0000000, -0.4480736]])
        tsr_R = TSR.rot_to_rpy(ninety_y_rot)
        cairo_R = rot2rpy(ninety_y_rot)
        np.testing.assert_array_almost_equal(tsr_R, cairo_R, decimal=4)
    
    def test_90_degree_z(self):
        ninety_z_rot = np.array([[ -0.4480736, -0.8939967,  0.0000000],
                        [0.8939967, -0.4480736,  0.0000000],
                        [0.0000000,  0.0000000,  1.0000000]])
        tsr_R = TSR.rot_to_rpy(ninety_z_rot)
        cairo_R = rot2rpy(ninety_z_rot)
        np.testing.assert_array_almost_equal(tsr_R, cairo_R, decimal=4)
    
    def test_90_degree_xyz(self):
        ninety_xyz_rot = np.array([[0.2007700,  0.4005763,  0.8939967],
                    [-0.7586902, -0.5137390,  0.4005763],
                    [0.6197423, -0.7586902,  0.2007700]])
        tsr_R = TSR.rot_to_rpy(ninety_xyz_rot)
        cairo_R = rot2rpy(ninety_xyz_rot)
        np.testing.assert_array_almost_equal(tsr_R, cairo_R, decimal=4)
