"""
Classes to represent robot Sensors in PyBullet simulations.
"""
import copy
import json
import os
from abc import abstractmethod

if os.environ.get('ROS_DISTRO'):
    import rospy
    from std_msgs.msg import Float32MultiArray, Float32
    from std_msgs.msg import String
import numpy as np
import pybullet as p


from cairo_simulator.core.simulator import Simulator, rosmethod
from cairo_simulator.core.utils import ASSETS_PATH, compute_3d_homogeneous_transform, multiply_quaternions

class Sensor():
    """
    Base class for Robot Sensors.
    """

    def __init__(self, sensor_name, robot_id=None, robot_link=None, position_offset=[0,0,0], orientation_offset=[0,0,0,1], fixed_pose=True, urdf_file=None, urdf_flags=0):
        '''
        Args:
            sensor_name (str): Identifier for this sensor
        
        Optional Args:
            robot_id (int): PyBullet id of robot this belongs to
            robot_link (int): Link id where this joint is 'mounted'
            position_offset (Tuple of 3 floats): X,Y,Z position of sensor relative to mount point (or world frame if robot_id is None)
            orientation_offset (Tuple of 4 floats): Quaternion orientation of sensor relative to mount point (or world frame if robot_id is None)
            fixed_pose (Bool): True if fixed in place, False if movable/subject to physics
            urdf_file (str): Location of the sensor's URDF file to render
            urdf_flags (int): PyBullet URDF flags to load with the model
        '''

        self._name = sensor_name
        self._simulator_id = None
        self._robot_id = robot_id
        self._robot_link = robot_link
        self._position_offset = np.array(position_offset) # Initial position, could change over the course of simulation!
        self._orientation_offset = np.array(orientation_offset) # Initial orientation, could change over the course of simulation!
        self._debug_mode = False # Set to True to trigger a Sensor's debugging mode (e.g., rendering lasers, line of sight, etc.)

        # If there's a physical model to render to represent this sensor, we have to care about physics
        if urdf_file is not None: 
            self._simulator_id = p.loadURDF(urdf_file, basePosition=position_offset, baseOrientation=orientation_offset, useFixedBase=fixed_pose, flags=urdf_flags)
            if self._robot_id is None:
                # Sensor exists independent of a robot
                p.createConstraint(self._simulator_id, -1, -1, -1, p.JOINT_FIXED, [0, 0, 0], [0, 0, 0], position_offset, orientation_offset)
            else:
                # Sensor is part of the robot, attach it to robot_id on its robot_link
                p.createConstraint(self._simulator_id, -1, self._robot_id, self._robot_link, p.JOINT_FIXED, [0, 0, 0], [0, 0, 0], position_offset, orientation_offset)
        
    @abstractmethod
    def get_position(self):
        if self._robot_id is None:
            return self._position_offset
        else:
            # Get position of robot link + offset
            link_state = p.getLinkState(self._robot_id, self._robot_link)
            position = p.multiplyTransforms(link_state[0], link_state[1], self._position_offset, self._orientation_offset)[0]
            return np.array(position)
    
    @abstractmethod
    def get_orientation(self):
        if self._robot_id is None:
            return self._orientation_offset
        else:
            # Get orientation of robot link + orientation offset
            link_state = p.getLinkState(self._robot_id, self._robot_link)
            orientation = p.multiplyTransforms(link_state[0], link_state[1], self._position_offset, self._orientation_offset)[1]
            return np.array(orientation)
    
    @abstractmethod
    def get_reading(self):
        pass

    @abstractmethod
    def set_debug_mode(self, bool_debug_enable):
        self._debug_mode = bool_debug_enable

class LaserRangeFinder(Sensor):
    '''
    Simple laser range finder sensor.
    Can be instantiated as part of a robot or freestanding.
    Debug mode (setting self._debug_mode=True) turns on rendering of the sensor's laser, which changes color if an object is detected.
    '''
    def __init__(self, sensor_name="LRF", robot_id=None, robot_link=None, position_offset=[0,0,0], orientation_offset=[0,0,0,1], fixed_pose=True):    
        super().__init__(sensor_name, robot_id=robot_id, robot_link=robot_link, position_offset=position_offset, orientation_offset=orientation_offset, fixed_pose=fixed_pose, urdf_file=None, urdf_flags=0)
        self._range = 10. # Default range of 10 meters
        self._debug_ray_list = []
        self._sensor_debug_id = p.addUserDebugLine(position_offset, position_offset, [0,0,0], lineWidth=10, lifeTime=0.01)

    def set_range(self, min_range, max_range):
        '''
        Args:
            min_range (float): Minimum detection range in meters from center of sensor
            max_range (float): Maximum detection range in meters from center of sensor
        '''
        self._max_range = max_range
        self._min_range = min_range
        self._range = max_range - min_range

    def get_reading(self):
        '''
        Find distance to nearest object in line-of-sight between [min_range, max_range] of th sensor. 
        Returns np.infty if none, otherwise returns distance in simulator units.
        '''

        from_pos = np.array(p.multiplyTransforms(self.get_position(), self.get_orientation(),[self._min_range,0,0], [0,0,0,1])[0])
        to_pos = np.array(p.multiplyTransforms(self.get_position(), self.get_orientation(),[self._max_range,0,0], [0,0,0,1])[0])

        # rayTest returns ((obj_id, link_index, percentage along ray of first intersection, hit position, hit normal vector))
        hit = p.rayTest(from_pos, to_pos)[0]

        if self._debug_mode is True:
            sensor_direction = np.array(p.multiplyTransforms(from_pos, self.get_orientation(),[0.01,0,0], [0,0,0,1])[0])

            ray_color = [0,0,1]
            if hit[0] == -1: ray_color = [0,1,0]
            p.addUserDebugLine(from_pos, to_pos, ray_color,lineWidth=3,lifeTime=0.1)
            self._sensor_debug_id = p.addUserDebugLine(from_pos, sensor_direction, [0,0,0], lineWidth=10, replaceItemUniqueId=self._sensor_debug_id)
        
        if hit[0] == -1: return np.infty
        return self._range * hit[2] + self._min_range # hit[2] is the % along the ray (from min to max range away from the sensor)
