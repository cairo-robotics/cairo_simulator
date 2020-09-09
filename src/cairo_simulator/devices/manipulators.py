"""
Classes to represent robot Manpulators in PyBullet simulations.
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
from ikpy.chain import Chain
from ikpy.urdf.URDF import get_chain_from_joints

from cairo_simulator.core.simulator import Simulator, Robot, rosmethod
from cairo_simulator.core.utils import ASSETS_PATH, compute_3d_homogeneous_transform, quaternion_from_matrix


class Manipulator(Robot):

    """
    Base class for Robot Manipulators with linked/articulated chains.
    """

    def __init__(self, robot_name, urdf_file, position, orientation=[0, 0, 0, 1], fixed_base=1, urdf_flags=0):
        """
        Initialize a Robot at coordinates position=[x,y,z] and add it to the simulator manager

        Args:
            robot_name (str): Name of the robot
            urdf_file (str): Filepath to urdf file.
            position (list): Point [x,y,z]
            orientation (list): Quaternion [x,y,z,w]
            fixed_base (int): 0 if base can be moved, 1 if fixed in place
            urdf_flags (int): Bitwise flags.
        """
        super().__init__(robot_name, urdf_file, position, orientation=orientation,
                         fixed_base=fixed_base, urdf_flags=urdf_flags)  # p.URDF_MERGE_FIXED_LINKS)

        if Simulator.using_ros():
            self._sub_position_update = rospy.Subscriber(
                '/%s/move_to_joint_pos' % self._name, Float32MultiArray, self.move_to_joint_pos_callback)
            self._sub_position_vel_update = rospy.Subscriber(
                '/%s/move_to_joint_pos_vel' % self._name, Float32MultiArray, self.move_to_joint_pos_vel_callback)
            self._sub_execute_trajectory = rospy.Subscriber(
                '/%s/execute_trajectory' % self._name, String, self.execute_trajectory_callback)

        # self._ik_service = rospy.Service('/%s/ik_service', Pose, self.ik_service)

        # Record joint indices of controllable DoF from PyBullet's loaded model.
        self._arm_dof_indices = []
        self._gripper_dof_indices = []

        # Initialize joint position/velocity limit data structures
        self._arm_joint_limits = []
        self._arm_joint_velocity_max = []  # Max velocity for each arm joint
        # Default velocity for moving the robot's joints
        self._arm_joint_default_velocity = []
        # List of indices of arm DoF within list of all non-fixed joints
        self._arm_ik_indices = []

        self._gripper_joint_limits = []
        self._gripper_joint_velocity_max = []  # Max velocity for each gripper joint
        # Default velocity for moving the gripper's joints
        self._gripper_joint_default_velocity = []

    def _init_local_vars(self):
        '''
        Function that should be called to initialize any local variables that couldn't be done in the Manipulator constructor
        '''
        self._init_joint_names()
        self._init_joint_limits()
        # Default to 50% of max movement speed
        self.set_default_joint_velocity_pct(0.5)

    @abstractmethod
    def _init_forward_kinematics(self, urdf_file):
        pass

    @abstractmethod
    def _init_joint_names(self):
        pass

    @abstractmethod
    def _init_joint_limits(self):
        pass

    @abstractmethod
    def publish_state(self):
        pass

    @abstractmethod
    def set_default_joint_velocity_pct(self, pct):
        pass

    @abstractmethod
    def move_to_joint_pos(self, target_position):
        '''
        Move arm to a target position

        Args:
            target_position (list): List of floats, indicating joint positions for the manipulator
        '''
        pass

    @abstractmethod
    def move_to_joint_pos_with_vel(self, target_position, target_velocity):
        '''
        Abtract method to move arm to a target position (interpolate) at a given velocity

        Args:
            target_position (): List of floats, indicating joint positions for the manipulator
            target_velocity (list): List of floats, indicating the speed that each joint should move to reach its position.
        '''
        pass

    @abstractmethod
    def get_current_joint_states(self):
        '''
        Returns a vector of the robot's joint positions.
        '''
        pass

    """
    def ik_service(self, req):
        '''
        ROS Service Wrapper for self.solve_inverse_kinematics
        Pass (0,0,0,0) for orientation to solve for position only.
        '''
        target_position = (req.position.x, req.position.y, req.position.z)
        target_orientation = (
            req.orientation.x, req.orientation.y, req.orientation.z, req.orientation.w)
        if target_orientation == (0,0,0,0): target_orientation = None

        soln = self.solve_inverse_kinematics(
            target_position, target_orientation)
        resp = Float32MultiArray()
        resp.data = soln
        return resp
    """

    @abstractmethod
    def get_jacobian(self, q):
        '''
        Method to get the the Jacobian of the given configuration q.
        '''
        pass

    def solve_forward_kinematics(self, joint_configuration):
        '''
        Returns the pose (point,quaternion) of the robot's end-effector given the provided joint_configuration
        in both world coords and local inertial frame coords.

        Args:
            joint_configuration (list): Vector of motor positions

        Returns:
            List: [(world_pos, world_ori), (local_pos, local_ori)]
        '''
        fk_results = self.fk_chain.forward_kinematics(
            joints=[0] * 2 + list(joint_configuration) + 3 * [0], full_kinematics=False)

        local_pos = list(fk_results[:3, 3])
        local_ori = quaternion_from_matrix(fk_results[:3, :3])
        base_pose, _ = p.getBasePositionAndOrientation(self._simulator_id)

        world_pos = [sum(x) for x in zip(local_pos, base_pose)]
        world_ori = local_ori
        return ((world_pos, world_ori), (local_pos, local_ori))

    def get_joint_pose_in_world_frame(self, joint_index=None):
        '''
        Returns the pose (point,quaternion) of the robot's joint at joint_index in the world frame. Defaults to end effector.

        Args:
            joint_index (None, optional): Index of the joint for which to get the pose.

        Returns:
            list, list: world_pos, world_quaternion
        '''
        if joint_index is None:
            joint_index = self._end_effector_link_index

        # In PyBullet, Link Index = Joint Index
        link_state = p.getLinkState(self._simulator_id, joint_index)

        world_pos = link_state[0]
        world_quaternion = link_state[1]

        return world_pos, world_quaternion

    def solve_inverse_kinematics(self, target_position, target_orientation=None, target_in_local_coords=False):
        '''
        Returns a robot configuration (list of joint positions) that gets the robot's end_effector_link
        as close as possible to target_position at target_orientation in world coordinates.

        If target_in_local_coords is true, then target_position is assumed to be from the robot's inertial frame.
        Wraps PyBullet's calculateInverseKinematics function.


        Args:
            target_position (list): List with target [x,y,z] coordinate.
            target_orientation (list/None, optional): List with target orientation either as Euler angles [r,p,y] or a quaternion [x,y,z,w].
            target_in_local_coords (bool, optional): Boolean flag to indicate whether to use robot's intertial frame.

        Returns:
            list: List of joint configurations.
        '''
        if self._end_effector_link_index == -1:
            rospy.logerr(
                "Inverse Kinematics solver not initialized properly for robot %s: end effector link index not set!" % self._name)
            return

        ik_solution = None

        if target_in_local_coords is True:
            # Convert position from robot-centric coordinate space to world coordinates
            robot_world_pose = p.getBasePositionAndOrientation(
                self._simulator_id)
            robot_world_position, robot_world_ori_euler = robot_world_pose[:3], p.getEulerFromQuaternion(
                robot_world_pose[3:])
            transform_from_robot_local_coord_to_world_frame = compute_3d_homogeneous_transform(
                robot_world_pose[0], robot_world_pose[1], robot_world_pose[2], robot_world_ori_euler[0], robot_world_ori_euler[1], robot_world_ori_euler[2])
            target_point = np.array([*target_position, 1])
            target_position = np.matmul(
                transform_from_robot_local_coord_to_world_frame, target_point.T)[:3]

        if target_orientation is None:
            ik_solution = p.calculateInverseKinematics(
                self._simulator_id, self._end_effector_link_index, target_position, maxNumIterations=120)
        else:
            if len(target_orientation) == 3:
                target_orientation = p.getQuaternionFromEuler(
                    target_orientation)
            ik_solution = p.calculateInverseKinematics(
                self._simulator_id, self._end_effector_link_index, target_position, targetOrientation=target_orientation, maxNumIterations=120)

        # Return a configuration of only the arm's joints.
        arm_config = [0] * len(self._arm_ik_indices)
        for i, idx in enumerate(self._arm_ik_indices):
            arm_config[i] = ik_solution[idx]

        return arm_config

    def execute_trajectory(self, trajectory_data):
        '''
        Execute a trajectory with the manipulator given positions and timings.
        This function computes the velocities needed to make the timeline.
        Ex: trajectory_data = [(1., [0,0,0,0,0,0,0]), (2.5, [1,0,0,0,0,0,0]), (4, [1,0,2.9,1.23,1.52,0,0])]
            Sends robot to 3 waypoints over the course of 4 seconds

        Args:
            trajectory_data (list): Vector of (time, joint configuration) tuples, indicating which joint positions the robot should achieve at which times. Set time=0 for each waypoint if you don't care about timing. Joint configuration vector contents should correspond to the parameters that work for for move_to_joint_pos
        '''
        joint_count = (len(self._arm_dof_indices) +
                       len(self._gripper_dof_indices))

        joint_positions = []
        joint_velocities = []

        # Initial robot position
        last_time = 0
        last_position = self.get_current_joint_states()

        joint_positions = [copy.copy(last_position)]
        joint_velocities = [[0] * joint_count]

        for waypoint in trajectory_data:
            target_duration = waypoint[0] - last_time
            target_position = waypoint[1]
            target_velocities = []

            if target_duration < 0.001:
                target_duration = 0.001  # No duration given, avoid divide-by-zero and move quickly

            # Compute distance from current position, compute per-joint velocity to reach in (t - t_{-1}) seconds
            # Each waypoint should have joint_count values
            if len(target_position) != joint_count:
                self.logger.warn("Bad trajectory waypoint passed to Manipulator %s. Had length %d. Aborting trajectory." % (
                    self._name, len(target_position)))
                return

            # Arm + Gripper velocity
            max_velocities = self._arm_joint_velocity_max + self._gripper_joint_velocity_max
            for i in range(len(target_position)):
                distance_to_cover = abs(target_position[i] - last_position[i])
                velocity = min(distance_to_cover /
                               target_duration, max_velocities[i])
                target_velocities.append(velocity)

            joint_positions.append(target_position)
            joint_velocities.append(target_velocities)

            last_time = waypoint[0]
            last_position = target_position  # 9-DoF arm+gripper position vector

        # Now that joint_positions and joint_velocities are populated, we can execute the trajectory
        sim = Simulator.get_instance()
        sim.set_robot_trajectory(
            self._simulator_id, joint_positions, joint_velocities)

    def check_if_at_position(self, pos, epsilon=0.001):
        '''
        Returns True if the robot's joints are within epsilon of pos, false otherwisen

        Args:
            pos (list): Vector of joint positions
            epsilon (float, optional): Distance threshold for 'at position'. Larger is more permissive.

        Returns:
            bool: Whether or not if at position within epsilon.
        '''

        joint_count = (len(self._arm_dof_indices) +
                       len(self._gripper_dof_indices))

        if len(pos) != joint_count:
            self.logger.warn("Invalid position given to check_if_at_position.")
            return False

        cur_pos = self.get_current_joint_states()

        return np.linalg.norm(np.array(pos) - np.array(cur_pos)) < epsilon


class Sawyer(Manipulator):

    """
    Concrete Manipulator representing a Sawyer Robot in Simulation.
    """

    def __init__(self, robot_name, position, orientation=[0, 0, 0, 1], fixed_base=0, publish_full_state=False):
        """
        Initialize a Sawyer Robot at coordinates (x,y,z) and add it to the simulator manager

        Args:
            robot_name (str): Name of the robot
            urdf_file (str): Filepath to urdf file.
            position (list): Point [x,y,z]
            orientation (list): Quaternion [x,y,z,w]
            fixed_base (int): 0 if base can be moved, 1 if fixed in place
            urdf_flags (int): Bitwise flags.
            publish_full_state (bool): True will publish more detailed state info., False will publish config/pose only.
        """
        super().__init__(robot_name, ASSETS_PATH +
                         'sawyer_description/urdf/sawyer_static.urdf', position, orientation, fixed_base)

        if Simulator.using_ros():
            # Should the full robot state be published each cycle (pos/vel/force), or just joint positions
            self._publish_full_state = publish_full_state
            self._pub_robot_state_full = rospy.Publisher(
                '/%s/robot_state_full' % self._name, String, queue_size=0)
            self._sub_head_pan = rospy.Subscriber(
                '/%s/set_head_pan' % self._name, Float32, self.set_head_pan)

        self._init_local_vars()
        self._init_forward_kinematics(
            ASSETS_PATH + 'sawyer_description/urdf/sawyer_static.urdf')

    def _init_forward_kinematics(self, urdf_file):
        gripper_tip_elements = get_chain_from_joints(urdf_file, joints=['right_arm_mount', 'right_j0', 'right_j1', 'right_j2',
                                                                        'right_j3', 'right_j4', 'right_j5', 'right_j6', 'right_hand', 'right_gripper_base_joint', 'right_gripper_tip_joint'])
        self.fk_chain = Chain.from_urdf_file(
            urdf_file, base_elements=gripper_tip_elements, active_links_mask=[True] + 8 * [True] + 3 * [False])

    def _init_joint_names(self):
        """
        Initialize joint names i.e.  Sawyer's"right_j0"
        """
        self._arm_dof_names = ['right_j0', 'right_j1', 'right_j2',
                               'right_j3', 'right_j4', 'right_j5', 'right_j6']
        self._gripper_dof_names = [
            'right_gripper_l_finger_joint', 'right_gripper_r_finger_joint']
        self._extra_dof_names = ['head_pan']

        # From base to wrist, j0 through j6 of Sawyer arm
        self._arm_dof_indices = self._populate_dof_indices(self._arm_dof_names)
        self._gripper_dof_indices = self._populate_dof_indices(
            self._gripper_dof_names)  # Left finger, Right finger
        self._extra_dof_indices = self._populate_dof_indices(
            self._extra_dof_names)

        # Find index of each arm DoF when only counting non-fixed joints for IK calls
        # Detail: IK solver returns a vector including positions for all non-fixed joints, we need to track which ones are part of the arm.
        self._arm_ik_indices = []
        actuated_joints = []
        for i in range(p.getNumJoints(self._simulator_id)):
            j_info = p.getJointInfo(self._simulator_id, i)
            if j_info[2] != p.JOINT_FIXED:
                actuated_joints.append(j_info[1].decode('UTF-8'))

        for joint_name in self._arm_dof_names:
            self._arm_ik_indices.append(actuated_joints.index(joint_name))

        self._end_effector_link_index = self._arm_dof_indices[-1]

    def _init_joint_limits(self):
        """
        Initialize join limits.
        """
        self._arm_joint_limits = [
        ]  # Seven elements, j0 through j6, containing a tuple with the (min,max) value
        self._arm_joint_velocity_max = []  # Max velocity for each arm joint
        self._arm_joint_default_velocity = []  # Default velocity for each arm joint

        self._gripper_joint_limits = []  # Same as arm, but for left and right finger
        self._gripper_joint_velocity_max = []  # Max velocity for each gripper joint
        # Default velocity for each gripper joint
        self._gripper_joint_default_velocity = []

        self._extra_joint_limits = []  # Head pan DOF
        self._extra_joint_velocity_max = []  # Head pan DOF
        # Default velocity for moving the robot's joints
        self._extra_joint_default_velocity = []

        # TODO: Modularize into inherited abstract function so all arms have the same setup structure
        for i in self._arm_dof_indices:
            joint_info = p.getJointInfo(self._simulator_id, i)
            self._arm_joint_limits.append((joint_info[8], joint_info[9]))
            self._arm_joint_velocity_max.append(joint_info[11])
        for i in self._gripper_dof_indices:
            joint_info = p.getJointInfo(self._simulator_id, i)
            self._gripper_joint_limits.append((joint_info[8], joint_info[9]))
            self._gripper_joint_velocity_max.append(joint_info[11])
        for i in self._extra_dof_indices:
            joint_info = p.getJointInfo(self._simulator_id, i)
            self._extra_joint_limits.append((joint_info[8], joint_info[9]))
            self._extra_joint_velocity_max.append(joint_info[11])

    def set_default_joint_velocity_pct(self, pct):
        '''
        Sets the default movement speed for each joint as a percentage of its maximum velocity.

        Args:
            pct (float): The percentage of max velocity.
        '''
        pct = max(min(1., pct), 0.)

        self._arm_joint_default_velocity = []
        for max_vel in self._arm_joint_velocity_max:
            self._arm_joint_default_velocity.append(max_vel * pct)

        self._gripper_joint_default_velocity = []
        for max_vel in self._gripper_joint_velocity_max:
            self._gripper_joint_default_velocity.append(max_vel * pct)

        self._extra_joint_default_velocity = []
        for max_vel in self._extra_joint_velocity_max:
            self._extra_joint_default_velocity.append(max_vel * pct)

    def set_head_pan(self, target_position, target_velocity=None):
        """
        Sets the Sawyer's tablet head to a given position at a given speed.

        Args:
        target_position (float): Target head position
        target_velocity (float): Desired motor velocity, Use None for default speed.
        """
        target_position = max(self._extra_joint_limits[0][0], min(
            val, self._extra_joint_limits[0][1]))
        if target_velocity is None:
            target_velocity = self._extra_joint_default_velocity[0]
        p.setJointMotorControl2(self._simulator_id, self._extra_dof_indices[0], p.POSITION_CONTROL,
                                target_position, target_velocity, maxVelocity=target_velocity)

    @rosmethod
    def publish_state(self):
        """
        Publish robot state if using ROS. 

        This will populate a state vector and publish the the robot state topic initalized during instantiation.
        """
        base_pose = p.getBasePositionAndOrientation(self._simulator_id)
        arm_configuration = []
        gripper_configuration = []

        arm_velocities = []
        # Fx, Fy, Fz, Mx, My, Mz  (Linear and rotational forces on joint)
        arm_forces = []
        gripper_velocities = []
        gripper_forces = []

        joint_states = p.getJointStates(
            self._simulator_id, self._arm_dof_indices)
        for joint in joint_states:
            arm_configuration.append(joint[0])
            arm_velocities.append(joint[1])
            arm_forces.append(joint[2])

        joint_states = p.getJointStates(
            self._simulator_id, self._gripper_dof_indices)
        for joint in joint_states:
            gripper_configuration.append(joint[0])
            gripper_velocities.append(joint[1])
            gripper_forces.append(joint[2])

        if self._publish_full_state is True:
            robot_state = {'base': base_pose, 'arm': {}, 'gripper': {}}
            robot_state['arm']['configuration'] = arm_configuration
            robot_state['arm']['velocities'] = arm_velocities
            robot_state['arm']['forces'] = arm_forces
            robot_state['gripper']['configuration'] = gripper_configuration
            robot_state['gripper']['velocities'] = gripper_velocities
            robot_state['gripper']['forces'] = gripper_forces

            self._pub_robot_state_full.publish(String(json.dumps(robot_state)))

        # TODO: Address difference between self._pub_robot_state and self._pub_robot_state_full
        # state_vector = Float32MultiArray()
        # state_vector.data = arm_configuration + gripper_configuration
        # self._pub_robot_state.publish(state_vector)

    def move_to_joint_pos_callback(self, target_position_float32array):
        """
        Moves to target joint position, will wait for current execution to finish.

        Args:
            target_position_float32array (list): List for float value sfor position.

        """
        if self._executing_trajectory:
            self.logger.warn(
                "Current trajectory for %s not finished executing, but new joint position received!" % self._name)
        return self.move_to_joint_pos(target_position_float32array.data)

    def move_to_joint_pos(self, target_position):
        '''
        Move Sawyer arm to a target position (interpolate)
        @param target_position: List of 7, 8, or 9 floats, indicating either 7: joint positions for Sawyer's 7 DoF arm, 8: joint positions for the 7 DoF arm and a percentage indicating how open the gripper should be, or 9: joint positions for the 7 DoF arm and positions for the 2 DoF gripper's finger positions

        Args:
            target_position (TYPE): The target description the length of which is depdendent on whether gripper open/close control is desired.
        '''
        list_tgt_position = list(target_position)
        if len(target_position) == 7:
            self.move_to_joint_pos_with_vel(
                list_tgt_position, self._arm_joint_default_velocity)
            # p.setJointMotorControlArray(self._simulator_id, self._arm_dof_indices, p.POSITION_CONTROL, targetPositions=target_position)
        elif len(target_position) == 8:
            self.move_to_joint_pos_with_vel(list_tgt_position + self.get_gripper_pct_finger_positions(
                target_position[7]), self._arm_joint_default_velocity + self._gripper_joint_default_velocity)
            # p.setJointMotorControlArray(self._simulator_id, self._arm_dof_indices + self._gripper_dof_indices, p.POSITION_CONTROL, targetPositions=target_position[:7] + self.get_gripper_pct_finger_positions(target_position[7]))
        elif len(target_position) == 9:
            self.move_to_joint_pos_with_vel(
                list_tgt_position, self._arm_joint_default_velocity + self._gripper_joint_default_velocity)
            # p.setJointMotorControlArray(self._simulator_id, self._arm_dof_indices + self._gripper_dof_indices, p.POSITION_CONTROL, targetPositions=target_position)
        else:
            self.logger.warn(
                "Invalid joint configuration provided for Sawyer %s. Needs to be 7 floats (arm) or 9 floats (arm+gripper)" % self._name)

    def move_to_joint_pos_vel_callback(self, target_position_vel_float32array):
        """
        Moves robot to joint position with given joint velocities.

        Args:
            target_position_vel_float32array (list): 9 positional and 9 velocity float values

        Returns:
            TYPE: Description
        """
        if len(target_position_vel_float32array.data) != 18:
            self.logger.warn(
                "Invalid position and velocity configuration provided for Sawyer %s. Must have 18 floats for 9 position and 9 velocity targets." % self._name)
            return
        return self.move_to_joint_pos_with_vel(target_position_vel_float32array.data[:9], target_position_vel_float32array.data[9:])

    def move_to_joint_pos_with_vel(self, desired_position, desired_velocity):
        '''
        Move Sawyer arm to a target position (interpolate) at a given velocity
        @param target_position: Vector of 9 floats, indicating joint positions for the 7 DoF arm and positions for the 2 DoF gripper's finger positions
        @param target_velocities: Vector of 9 floats, indicating the speed that each joint should move to reach its position.

        Args:
            desired_position (list): Desired position/joint positions
            desired_velocity (list): Desired joint velocities.

        '''

        target_position = list(desired_position)
        target_velocity = list(desired_velocity)

        joints_list = self._arm_dof_indices + self._gripper_dof_indices
        if len(target_velocity) != len(target_position):
            rospy.logwarn("Different sizes of target positions (%d) and velocities (%d) passed to move_to_joint_pos_with_vel!" % (
                len(target_position), len(target_velocity)))
            return

        elif len(target_position) != 7 and len(target_position) != 9:
            rospy.logwarn("Invalid joint configuration/velocities provided for Sawyer %s. Function requires lists to be of length 1-7 floats (arm DoF only) or 9 floats (7 arm + 2 gripper DoF)" % self._name)

        elif len(target_position) == 7:
            gripper_pos = p.getJointStates(
                self._simulator_id, self._gripper_dof_indices)
            for entry in gripper_pos:
                target_position.append(entry[0])
                target_velocity.append(entry[1])
        '''
        tgt_positions = []
        tgt_velocities = []
        joint_states = p.getJointStates(self._simulator_id, range(p.getNumJoints(self._simulator_id)))
        for entry in joint_states:
            tgt_positions.append(entry[0])
            tgt_velocities.append(entry[1])

        for i, idx in enumerate(self._arm_dof_indices):
            tgt_positions[idx] = target_position[i]
            tgt_velocities[idx] = target_velocity[i]

        dynamics = p.calculateInverseDynamics(self._simulator_id, tgt_positions, tgt_velocities, [0]*len(tgt_positions))
        '''
        # p.setJointMotorControlArray(self._simulator_id, joints_list[:len(target_positions)], p.POSITION_CONTROL, targetPositions=target_position, targetVelocities=target_velocity)

        for i, j_idx in enumerate(joints_list):
            p.setJointMotorControl2(self._simulator_id, j_idx, p.POSITION_CONTROL,
                                    target_position[i], target_velocity[i], maxVelocity=target_velocity[i])

    def move_with_joint_vel(self, desired_vel):
        """
        Move sawyer joints with desired velocity.
        NOTE does not check against max vel and only uses main 7DOF arm
        """

        target_velocity = list(desired_vel)
        joints_list = self._arm_dof_indices

        if len(joints_list) is not len(desired_vel):
            self.logger.warn("wrong size torque list")
            return

        for i, j_idx in enumerate(joints_list):
            p.setJointMotorControl2(self._simulator_id,
                                    j_idx,
                                    p.VELOCITY_CONTROL,
                                    targetVelocity=target_velocity[i])

    def get_current_joint_states(self):
        """
        Returns the current joint states.

        Returns:
            list: List of joint angles / configuration.
        """
        position = []
        joint_states = p.getJointStates(
            self._simulator_id, self._arm_dof_indices + self._gripper_dof_indices)
        for entry in joint_states:
            position.append(entry[0])
        return position
    
    def get_jacobian(self, q, link_id=16, com_trans=(-8.0726e-06, 0.0085838, -0.0049566), vel_vec=[0.0]*7, accel_vec=[0.0]*7):
        # infill extra DOF around arm jonts (e.g. head pan, gripper etc,.)
        q = np.insert(q, [1], 0.0)
        q = np.append(q, [0.0, 0.0])
        vel_vec = np.insert(vel_vec, [1], 0.0)
        vel_vec = np.append(vel_vec, [0.0, 0.0])
        accel_vec = np.insert(accel_vec, [1], 0.0)
        accel_vec = np.append(accel_vec, [0.0, 0.0])
        # q = [0.7876748441700757, 0.0, 0.6013938842119559, 0.8830634552952313, 0.8964338093527918, 1.0725899856959438, 1.0725862705452007, 1.0946949885092667, 0.0, 0.0]
        vel_vec = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        accel_vec = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        client_id = Simulator.get_client_id()
        jac_t, jac_r = p.calculateJacobian(
            self._simulator_id, link_id, list(com_trans), list(q), list(vel_vec), list(accel_vec), client_id)
        J = np.vstack([np.array(jac_t), np.array(jac_r)])[
            :, [0, 2, 3, 4, 5, 6, 7]]
        return J

    def execute_trajectory_callback(self, trajectory_json_string):
        """
        Executed the given trajectory encoded as in a predefined JSON format.

        Args:
            trajectory_json_string (str): JSON formatted string.
        """
        if self._executing_trajectory:
            self.logger.warn(
                "Current trajectory for %s not finished executing, but new trajectory received!" % self._name)
        traj_data = json.loads(trajectory_json_string.data)
        self.execute_trajectory(traj_data)

    def execute_trajectory(self, trajectory_data):
        '''
        Execute a trajectory with the Sawyer arm given positions and timings. This function computes the velocities needed to make the timeline.
        Ex: trajectory_data = [(1., [0,0,0,0,0,0,0]), (2.5, [1,0,0,0,0,0,0,0]), (4, [1,0,2.9,1.23,1.52,0,0,0])]
            Sends robot to 3 waypoints over the course of 4 seconds
        @param trajectory_data Vector of (time, joint configuration) tuples, indicating which joint positions the robot should achieve at which times. Set time=0 for each waypoint if you don't care about timing. Joint configuration vectors can be 7, 8, or 9 floats corresponding to the parameter for move_to_joint_pos (7: arm only, 8: arm + gripper %open, 9: arm + gripper finger positions)

        Args:
            trajectory_data (list): List of tuples (time, configuration list)
        '''

        joint_positions = []
        joint_velocities = []

        # Initial robot position
        last_time = 0
        last_position = self.get_current_joint_states()

        joint_positions = [copy.copy(last_position)]
        joint_velocities = [[0] * 9]

        for waypoint in trajectory_data:
            target_duration = waypoint[0] - last_time
            target_position = waypoint[1]
            target_velocities = []

            if target_duration < 0.001:
                target_duration = 0.001  # No duration given, avoid divide-by-zero and move quickly

            # Compute distance from current position, compute per-joint velocity to reach in (t - t_{-1}) seconds
            # Each waypoint should have 7-9 values
            if len(target_position) < 7 or len(target_position) > 9:
                self.logger.warn("Bad trajectory waypoint passed to Sawyer %s. Had length %d. Aborting trajectory." %
                                 (self._name, len(target_position)))
                return

            # target_position will be 9-DoF vector for arm+gripper position after this code block
            if len(target_position) == 7:
                # Keep old gripper position
                target_position = target_position[:7] + last_position[7:9]
            elif len(target_position) == 8:  # Arm + Gripper %
                next_pos_gripper = self.get_gripper_pct_finger_positions(
                    target_position[7])
                target_position = target_position[:7] + next_pos_gripper

            # Arm + Gripper velocity
            max_velocities = self._arm_joint_velocity_max + self._gripper_joint_velocity_max
            for i in range(len(target_position)):
                distance_to_cover = abs(target_position[i] - last_position[i])
                velocity = min(distance_to_cover /
                               target_duration, max_velocities[i])
                target_velocities.append(velocity)

            joint_positions.append(target_position)
            joint_velocities.append(target_velocities)

            last_time = waypoint[0]
            last_position = target_position  # 9-DoF arm+gripper position vector

        # Now that joint_positions and joint_velocities are populated, we can execute the trajectory
        sim = Simulator.get_instance()
        sim.set_robot_trajectory(
            self._simulator_id, joint_positions, joint_velocities)

    def check_if_at_position(self, pos, epsilon=0.2):
        '''
        Returns True if the robot's joints are within (epsilon) of pos, false otherwise
        @param pos Vector of length 7, 8, or 9, corresponding to arm position, arm+gripper%, or arm+gripper position

        Args:
            pos (list): Vector of length 7, 8, or 9, corresponding to arm position, arm+gripper%, or arm+gripper position
            epsilon (float, optional): Within bounds

        Returns:
            bool: True if within epsilon ball, else False.
        '''
        if len(pos) < 7 or len(pos) > 9:
            self.logger.warn(
                "Invalid position given to check_if_at_position. Must be length 7, 8, or 9 for Sawyer.")
            return False

        cur_pos = self.get_current_joint_states()

        if len(pos) == 7:
            cur_pos = cur_pos[:7]
        elif len(pos) == 8:
            pos = pos[:7] + self.get_gripper_pct_finger_positions(pos[7])

        dist = np.linalg.norm(np.array(pos) - np.array(cur_pos))

        # rospy.loginfo("Checking if Sawyer is at %s. (dist=%g) %s" % (str(pos), dist, str(dist <= epsilon)))
        if dist <= epsilon:
            return True
        return False

    def get_gripper_pct_finger_positions(self, pct_gripper_open):
        '''
        Returns the target position of each gripper finger given a percentage of how open the gripper should be

        Args:
            pct_gripper_open (float): pct_gripper_open Value in range [0.,1.] describing how open the gripper should be

        Returns:
            float, float: The left and right position.
        '''
        pct_gripper_open = max(0., min(1., pct_gripper_open))
        max_displacement = 0
        for limit in self._gripper_joint_limits:
            max_displacement += limit[1] - limit[0]

        total_displacement = pct_gripper_open * max_displacement
        left_position = total_displacement / 2.
        right_position = total_displacement / -2.

        return left_position, right_position
