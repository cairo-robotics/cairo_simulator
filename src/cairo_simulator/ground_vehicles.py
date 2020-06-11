import copy
import json
import numpy as np
import os
import pybullet as p
import pybullet_data
from abc import abstractmethod
from .Simulator import Simulator
from .Simulator import Robot
from cairo_simulator import Utils

class GroundVehicle(Robot):
    def __init__(self, robot_name, urdf_file, position, orientation=[0,0,0,1], fixed_base=0, urdf_flags=0):
        """
        Initialize a Robot at coordinates position=(x,y,z) and add it to the simulator manager
        @param robot_name Internal identifier for this robot, used for ROS namespacing and debugging info
        @param urdf_file Path to URDF file
        @param position [x,y,z] position of the robot in the world
        @param orientation Quaternion [x,y,z,w] for the robot's start position. Default to upright.
        @param fixed_base Indicates if the base should be static in the world (0=False, 1=True)
        """
        super().__init__(robot_name, urdf_file, position, orientation=orientation, fixed_base=fixed_base, urdf_flags=urdf_flags) # p.URDF_MERGE_FIXED_LINKS)

        self._wheel_dof_indices = [] # Standardize tracking of locomotion-related DoFs
        self._steer_dof_indices = [] # Standardize tracking of steering-related DoFs

        self._wheel_dof_velocity_max = []
        self._steer_dof_velocity_max = []

    @abstractmethod
    def publish_state(self):
        pass



class Racecar(GroundVehicle):
    DRIVE_AWD = 0
    DRIVE_FWD = 1
    DRIVE_RWD = 2
    def __init__(self, robot_name, position, orientation=[0,0,0,1], drive_mode=None):
        """
        Initialize a Sawyer Robot at coordinates (x,y,z) and add it to the simulator manager
        """
        super().__init__(robot_name, os.path.join(pybullet_data.getDataPath(), 'racecar/racecar.urdf'), position, orientation)

        self._wheel_dof_names = ['left_front_wheel_joint', 'right_front_wheel_joint', 'left_rear_wheel_joint', 'right_rear_wheel_joint']
        self._steer_dof_names = ['left_steering_hinge_joint','right_steering_hinge_joint']

        # Set active vs. passive wheels depending on drive mode
        if drive_mode is None: drive_mode = Racecar.DRIVE_AWD
        self._drive_mode = drive_mode

        if self._drive_mode == Racecar.DRIVE_FWD:
            self._wheel_dof_indices = self._populate_dof_indices(self._wheel_dof_names[0:2])
            self._passive_wheel_indices = self._populate_dof_indices(self._wheel_dof_names[2:])
        elif self._drive_mode == Racecar.DRIVE_RWD:
            self._wheel_dof_indices = self._populate_dof_indices(self._wheel_dof_names[2:])
            self._passive_wheel_indices = self._populate_dof_indices(self._wheel_dof_names[0:2])
        else:
            self._wheel_dof_indices = self._populate_dof_indices(self._wheel_dof_names)
            self._passive_wheel_indices = []

        # Initialize joint position/velocity limits for active wheels and steering DoFs
        self._steer_dof_indices = self._populate_dof_indices(self._steer_dof_names)
        self._steer_joint_limits = []
        for steer_idx in self._steer_dof_indices:
            joint_info = p.getJointInfo(self._simulator_id, steer_idx)
            self._steer_joint_limits.append((joint_info[8], joint_info[9]))
            self._steer_dof_velocity_max.append(joint_info[11])

        for wheel_idx in self._wheel_dof_indices:
            joint_info = p.getJointInfo(self._simulator_id, steer_idx)
            self._wheel_dof_velocity_max.append(joint_info[11])

        for wheel_idx in self._passive_wheel_indices:
            p.setJointMotorControl2(self._simulator_id, wheel_idx, p.VELOCITY_CONTROL, targetVelocity=0, force=0)

    def publish_state(self):
        base_pose = p.getBasePositionAndOrientation(self._simulator_id)
        wheel_configuration = []
        wheel_velocities = []
        wheel_forces = []

        steering_configuration = []
        steering_velocities = []
        steering_forces = []

        joint_states = p.getJointStates(self._simulator_id, self._wheel_dof_indices)
        for joint in joint_states:
            wheel_configuration.append(joint[0])
            wheel_velocities.append(joint[1])
            wheel_forces.append(joint[2])

        joint_states = p.getJointStates(self._simulator_id, self._steering_dof_indices)
        for joint in joint_states:
            steering_configuration.append(joint[0])
            steering_velocities.append(joint[1])
            steering_forces.append(joint[2])

        if self._publish_full_state is True:
            robot_state = {'base': base_pose, 'wheels': {}, 'steering': {}}
            robot_state['wheels']['configuration'] = wheel_configuration
            robot_state['wheels']['velocities'] = wheel_velocities
            robot_state['wheels']['forces'] = wheel_forces
            robot_state['steering']['configuration'] = steering_configuration
            robot_state['steering']['velocities'] = steering_velocities
            robot_state['steering']['forces'] = steering_forces

            self._pub_robot_state_full.publish(String(json.dumps(robot_state)))

        state_vector = Float32MultiArray()
        state_vector.data = wheel_velocities + steering_configuration
        self._pub_robot_state.publish(state_vector)

    def set_steer_angle(self, target_position):
        '''
        Set the wheel angle to target_position
        '''
        target_position = max(min(self._steer_joint_limits[0][1], target_position), self._steer_joint_limits[0][0])
        for steer_idx in self._steer_dof_indices:
            p.setJointMotorControl2(self._simulator_id, steer_idx, p.POSITION_CONTROL, targetPosition=target_position)



    def set_wheel_velocities(self, target_velocities):
        '''
        Command active wheels (based on drive mode) to target velocities. Ordering is Front-left, front-right, rear-left, rear-right.
        @param target_velocities: Vector of 2-4 floats, indicating the speed that each wheel should move.
        '''
        
        for i in range(len(target_velocities)):
            target_velocities[i] = min(self._wheel_dof_velocity_max[i], max(-self._wheel_dof_velocity_max[i]),target_velocities[i])

        for i, wheel_idx in enumerate(self._wheel_dof_indices):
            p.setJointMotorControl2(self._simulator_id, wheel_idx, p.VELOCITY_CONTROL, targetVelocity=target_velocities[i], force=target_velocities[i])
