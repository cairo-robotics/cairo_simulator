import time
import copy
import json
import numpy as np
import rospy
import pybullet as p 
from std_msgs.msg import Float32MultiArray, Float32
from std_msgs.msg import String, Empty
from geometry_msgs.msg import Pose
from abc import abstractmethod
from .Simulator import ASSETS_PATH
from .Simulator import Simulator
from .Simulator import Robot

class Manipulator(Robot):
    def __init__(self, robot_name, urdf_file, x, y, z):   
        """
        Initialize a Robot at coordinates (x,y,z) and add it to the simulator manager
        """ 
        super().__init__(robot_name, urdf_file, x, y, z, p.URDF_MERGE_FIXED_LINKS)

        self._sub_position_update = rospy.Subscriber('/%s/move_to_joint_pos'%self._name, Float32MultiArray, self.move_to_joint_pos_callback)
        self._sub_position_vel_update = rospy.Subscriber('/%s/move_to_joint_pos_vel'%self._name, Float32MultiArray, self.move_to_joint_pos_vel_callback)
        self._sub_execute_trajectory = rospy.Subscriber('/%s/execute_trajectory'%self._name, String, self.execute_trajectory_callback)

        # self._ik_service = rospy.Service('/%s/ik_service', Pose, self.ik_service)

        self._end_effector_link_index = -1 # Must be set by instantiating class

        # Record joint indices of controllable DoF from PyBullet's loaded model.
        self._arm_dof_indices = []
        self._gripper_dof_indices = [] 

        # Initialize joint position/velocity limit data structures
        self._arm_joint_limits = [] 
        self._arm_joint_velocity_max = [] # Max velocity for each arm joint
        self._arm_joint_default_velocity = [] # Default velocity for moving the robot's joints

        self._gripper_joint_limits = [] 
        self._gripper_joint_velocity_max = [] # Max velocity for each gripper joint
        self._gripper_joint_default_velocity = [] # Default velocity for moving the gripper's joints


    def _init_local_vars(self):
        '''
        Function that should be called to initialize any local variables that couldn't be done in the Manipulator constructor
        '''
        self._init_joint_names()
        self._init_joint_limits()
        self.set_default_joint_velocity_pct(0.5) # Default to 50% of max movement speed

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
        @param target_position List of floats, indicating joint positions for the manipulator
        '''
        pass

    @abstractmethod
    def move_to_joint_pos_with_vel(self, target_position, target_velocity):
        '''
        Move arm to a target position (interpolate) at a given velocity
        @param target_position List of floats, indicating joint positions for the manipulator
        @param target_velocities: List of floats, indicating the speed that each joint should move to reach its position.
        '''
        pass

    @abstractmethod
    def get_current_joint_states(self):
        '''
        Returns a vector of the robot's joint positions
        '''
        pass

    """
    def ik_service(self, req):
        '''
        ROS Service Wrapper for self.solve_inverse_kinematics
        Pass (0,0,0,0) for orientation to solve for position only.
        '''
        target_position = (req.position.x, req.position.y, req.position.z)
        target_orientation = (req.orientation.x, req.orientation.y, req.orientation.z, req.orientation.w)
        if target_orientation == (0,0,0,0): target_orientation = None

        soln = self.solve_inverse_kinematics(target_position, target_orientation)
        resp = Float32MultiArray()
        resp.data = soln
        return resp
    """        
    def solve_forward_kinematics(self, joint_configuration, terminal_joint_index=None):
        '''
        Returns the pose (point,quaternion) of the robot's end-effector given the provided joint_configuration
        in both world coords and local inertial frame coords.
        @param joint_configuration Vector of motor positions
        @param terminal_joint_index Joint/Link index whose position is desired
        @return [(world_pos, world_ori), (local_pos, local_ori)]
        '''

        if terminal_joint_index is None:
            terminal_joint_index = self._end_effector_link_index

        # Store current joint state information
        cur_pos = p.getJointStates(self._simulator_id, self._arm_dof_indices)

        # Disable collisions
        #for idx in self._arm_dof_indices:
        #    p.setCollisionFilterGroupMask(self._simulator_id, idx, 0, 0)

        # Set new configuration and get link states
        for i, idx in enumerate(self._arm_dof_indices):
            p.resetJointState(self._simulator_id, idx, targetValue=joint_configuration[i], targetVelocity=0)

        link_state = p.getLinkState(self._simulator_id, terminal_joint_index)
        world_pos = link_state[0]
        world_ori = link_state[1]
        local_pos = link_state[2]
        local_ori = link_state[3]

        # Restore previous joint states
        for joint, idx in enumerate(self._arm_dof_indices):
            p.resetJointState(self._simulator_id, idx, targetValue=cur_pos[joint][0], targetVelocity=cur_pos[joint][1])


        return ((world_pos, world_ori), (local_pos, local_ori))

    def get_joint_pose_in_world_frame(self, joint_index=None):
        '''
        Returns the pose (point,quaternion) of the robot's joint at joint_index in the world frame. Defaults to end effector.
        '''
        if joint_index is None: joint_index = self._end_effector_link_index

        link_state = p.getLinkState(self._simulator_id, joint_index) #In PyBullet, Link Index = Joint Index
        
        world_pos = link_state[0]
        world_quaternion = link_state[1]
        
        return world_pos, world_quaternion


    def solve_inverse_kinematics(self, target_position, target_orientation=None, target_in_local_coords=False):
        '''
        Returns a robot configuration (list of joint positions) that gets the robot's end_effector_link 
        as close as possible to target_position at target_orientation in world coordinates.
        If target_in_local_coords is true, then target_position is assumed to be from the robot's inertial frame.
        Wraps PyBullet's calculateInverseKinematics function.

        @param target_position List with target [x,y,z] coordinate.
        @param target_orientation (Optional) List with target orientation either as Euler angles [r,p,y] or a quaternion [x,y,z,w].
        '''
        if self._end_effector_link_index == -1:
            rospy.logerr("Inverse Kinematics solver not initialized properly for robot %s: end effector link index not set!" % self._name)
            return

        ik_solution = None

        if target_in_local_coords is True:
            cur_pos = self.get_current_joint_states()[:len(self._arm_dof_indices)]
            if target_orientation is None:
                ik_solution = p.calculateInverseKinematics(self._simulator_id, self._end_effector_link_index, target_position, currentPosition=cur_pos, maxNumIterations=120)
            else:
                if len(target_orientation) == 3: target_orientation = p.getQuaternionFromEuler(target_orientation)
                ik_solution = p.calculateInverseKinematics(self._simulator_id, self._end_effector_link_index, target_position, targetOrientation=target_orientation, currentPosition=cur_pos, maxNumIterations=120)
        else:
            if target_orientation is None:
                ik_solution = p.calculateInverseKinematics(self._simulator_id, self._end_effector_link_index, target_position, maxNumIterations=120)
            else:
                if len(target_orientation) == 3: target_orientation = p.getQuaternionFromEuler(target_orientation)
                ik_solution = p.calculateInverseKinematics(self._simulator_id, self._end_effector_link_index, target_position, targetOrientation=target_orientation, maxNumIterations=120)

        # Return a configuration of only the arm's joints.
        arm_config = [0]*len(self._arm_dof_indices)
        for i, idx in enumerate(self._arm_dof_indices):
            arm_config[i] = ik_solution[idx]

        return arm_config        


    def execute_trajectory(self, trajectory_data):
        '''
        Execute a trajectory with the manipulator given positions and timings.
        This function computes the velocities needed to make the timeline.        
        Ex: trajectory_data = [(1., [0,0,0,0,0,0,0]), (2.5, [1,0,0,0,0,0,0]), (4, [1,0,2.9,1.23,1.52,0,0])]
            Sends robot to 3 waypoints over the course of 4 seconds
        @param trajectory_data Vector of (time, joint configuration) tuples, indicating which joint positions the robot should achieve at which times. Set time=0 for each waypoint if you don't care about timing. Joint configuration vector contents should correspond to the parameters that work for for move_to_joint_pos
        '''
        joint_count = (len(self._arm_dof_indices) + len(self._gripper_dof_indices))

        joint_positions = []
        joint_velocities = []

        # Initial robot position
        last_time = 0
        last_position = self.get_current_joint_states()

        joint_positions = [copy.copy(last_position)]
        joint_velocities = [[0]*joint_count]

        for waypoint in trajectory_data:
            target_duration = waypoint[0] - last_time
            target_position = waypoint[1]
            target_velocities = []

            if target_duration < 0.001:                
                target_duration = 0.001 # No duration given, avoid divide-by-zero and move quickly

            # Compute distance from current position, compute per-joint velocity to reach in (t - t_{-1}) seconds
            # Each waypoint should have joint_count values
            if len(target_position) != joint_count:
                rospy.logwarn("Bad trajectory waypoint passed to Manipulator %s. Had length %d. Aborting trajectory." % (self._name, len(target_position)))
                return

            # Arm + Gripper velocity
            max_velocities = self._arm_joint_velocity_max + self._gripper_joint_velocity_max
            for i in range(len(target_position)):
                distance_to_cover = abs(target_position[i] - last_position[i])
                velocity = min(distance_to_cover / target_duration, max_velocities[i])
                target_velocities.append(velocity)
            
            joint_positions.append(target_position)
            joint_velocities.append(target_velocities)

            last_time = waypoint[0]
            last_position = target_position # 9-DoF arm+gripper position vector

        # Now that joint_positions and joint_velocities are populated, we can execute the trajectory
        sim = Simulator.get_instance()
        sim.set_robot_trajectory(self._simulator_id, joint_positions, joint_velocities)
            
    def check_if_at_position(self, pos, epsilon=0.001):
        '''
        Returns True if the robot's joints are within epsilon of pos, false otherwise
        @param pos Vector of joint positions
        @param epsilon Distance threshold for 'at position'. Larger is more permissive.
        '''

        joint_count = (len(self._arm_dof_indices) + len(self._gripper_dof_indices))

        if len(pos) != joint_count: 
            rospy.logwarn("Invalid position given to check_if_at_position.")
            return False

        cur_pos = self.get_current_joint_states()

        return np.linalg.norm(np.array(pos) - np.array(cur_pos)) < epsilon


class Sawyer(Manipulator):
    def __init__(self, robot_name, x=0, y=0, z=0.8, publish_full_state=False):   
        """
        Initialize a Sawyer Robot at coordinates (x,y,z) and add it to the simulator manager
        """ 
        super().__init__(robot_name, ASSETS_PATH + 'sawyer_description/urdf/sawyer_static.urdf', x,y,z)

        self._publish_full_state = publish_full_state # Should the full robot state be published each cycle (pos/vel/force), or just joint positions
        self._pub_robot_state_full = rospy.Publisher('/%s/robot_state_full'%self._name, String, queue_size=0)
        self._sub_head_pan = rospy.Subscriber('/%s/set_head_pan'%self._name, Float32, self.set_head_pan)

        self._init_local_vars()

    def _init_joint_names(self):
        self._arm_dof_names = ['right_j0', 'right_j1', 'right_j2', 'right_j3', 'right_j4', 'right_j5', 'right_j6']
        self._gripper_dof_names = ['right_gripper_l_finger_joint', 'right_gripper_r_finger_joint']
        self._extra_dof_names = ['head_pan']

        self._arm_dof_indices = self._populate_dof_indices(self._arm_dof_names) # From base to wrist, j0 through j6 of Sawyer arm
        self._gripper_dof_indices = self._populate_dof_indices(self._gripper_dof_names) # Left finger, Right finger
        self._extra_dof_indices = self._populate_dof_indices(self._extra_dof_names)

        self._end_effector_link_index = self._arm_dof_indices[-1]

    def _init_joint_limits(self):
        # Initialize joint limits
        self._arm_joint_limits = [] # Seven elements, j0 through j6, containing a tuple with the (min,max) value
        self._arm_joint_velocity_max = [] # Max velocity for each arm joint
        self._arm_joint_default_velocity = [] # Default velocity for each arm joint

        self._gripper_joint_limits = [] # Same as arm, but for left and right finger
        self._gripper_joint_velocity_max = [] # Max velocity for each gripper joint
        self._gripper_joint_default_velocity = [] # Default velocity for each gripper joint

        self._extra_joint_limits = [] # Head pan DOF
        self._extra_joint_velocity_max = [] # Head pan DOF
        self._extra_joint_default_velocity = [] # Default velocity for moving the robot's joints

        # TODO: Modularize into inherited abstract function so all arms have the same setup structure
        for i in self._arm_dof_indices:
            joint_info = p.getJointInfo(self._simulator_id, i)
            self._arm_joint_limits.append( (joint_info[8], joint_info[9]) )
            self._arm_joint_velocity_max.append(joint_info[11])
        for i in self._gripper_dof_indices:
            joint_info = p.getJointInfo(self._simulator_id, i)
            self._gripper_joint_limits.append( (joint_info[8], joint_info[9]) )
            self._gripper_joint_velocity_max.append(joint_info[11])
        for i in self._extra_dof_indices:
            joint_info = p.getJointInfo(self._simulator_id, i)
            self._extra_joint_limits.append( (joint_info[8], joint_info[9]) )
            self._extra_joint_velocity_max.append(joint_info[11])

    def set_default_joint_velocity_pct(self, pct):
        '''
        Sets the default movement speed for each joint as a percentage of its maximum velocity.
        '''
        pct = max(min(1.,pct), 0.)

        self._arm_joint_default_velocity = []
        for max_vel in self._arm_joint_velocity_max:
            self._arm_joint_default_velocity.append(max_vel * pct)

        self._gripper_joint_default_velocity = []
        for max_vel in self._gripper_joint_velocity_max:
            self._gripper_joint_default_velocity.append(max_vel * pct)

        self._extra_joint_default_velocity = []
        for max_vel in self._extra_joint_velocity_max:
            self._extra_joint_default_velocity.append(max_vel * pct)


    def set_head_pan(self, val):
        target_position = max(self._extra_joint_limits[0][0], min(val.data, self._extra_joint_limits[0][1]))
        # TODO: Update to use setJointMotorControl with maxVelocity
        p.setJointMotorControlArray(self._simulator_id, self._extra_dof_indices, p.POSITION_CONTROL, targetPositions=target_position)


    def publish_state(self):
        base_pose = p.getBasePositionAndOrientation(self._simulator_id)
        arm_configuration = []
        gripper_configuration = []

        arm_velocities = []
        arm_forces = [] # Fx, Fy, Fz, Mx, My, Mz  (Linear and rotational forces on joint)
        gripper_velocities = []
        gripper_forces = []

        joint_states = p.getJointStates(self._simulator_id, self._arm_dof_indices)
        for joint in joint_states:
            arm_configuration.append(joint[0])
            arm_velocities.append(joint[1])
            arm_forces.append(joint[2])

        joint_states = p.getJointStates(self._simulator_id, self._gripper_dof_indices)
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

        state_vector = Float32MultiArray()
        state_vector.data = arm_configuration + gripper_configuration
        self._pub_robot_state.publish(state_vector)

    def move_to_joint_pos_callback(self, target_position_float32array):
        if self._executing_trajectory: rospy.logwarn("Current trajectory for %s not finished executing, but new joint position received!" % self._name)
        return self.move_to_joint_pos(target_position_float32array.data)

    def move_to_joint_pos(self, target_position):
        '''
        Move Sawyer arm to a target position (interpolate)
        @param target_position: List of 7, 8, or 9 floats, indicating either 7: joint positions for Sawyer's 7 DoF arm, 8: joint positions for the 7 DoF arm and a percentage indicating how open the gripper should be, or 9: joint positions for the 7 DoF arm and positions for the 2 DoF gripper's finger positions
        '''
        list_tgt_position = list(target_position)
        if len(target_position) == 7: 
            self.move_to_joint_pos_with_vel(list_tgt_position, self._arm_joint_default_velocity)
            #p.setJointMotorControlArray(self._simulator_id, self._arm_dof_indices, p.POSITION_CONTROL, targetPositions=target_position)
        elif len(target_position) == 8:
            self.move_to_joint_pos_with_vel(list_tgt_position + self.get_gripper_pct_finger_positions(target_position[7]), self._arm_joint_default_velocity + self._gripper_joint_default_velocity)
            #p.setJointMotorControlArray(self._simulator_id, self._arm_dof_indices + self._gripper_dof_indices, p.POSITION_CONTROL, targetPositions=target_position[:7] + self.get_gripper_pct_finger_positions(target_position[7]))
        elif len(target_position) == 9:
            self.move_to_joint_pos_with_vel(list_tgt_position, self._arm_joint_default_velocity + self._gripper_joint_default_velocity)
            #p.setJointMotorControlArray(self._simulator_id, self._arm_dof_indices + self._gripper_dof_indices, p.POSITION_CONTROL, targetPositions=target_position)
        else:
            rospy.logwarn("Invalid joint configuration provided for Sawyer %s. Needs to be 7 floats (arm) or 9 floats (arm+gripper)" % self._name)
            return

    def move_to_joint_pos_vel_callback(self, target_position_vel_float32array):
        if len(target_position_vel_float32array.data) != 18:
            rospy.logwarn("Invalid position and velocity configuration provided for Sawyer %s. Must have 18 floats for 9 position and 9 velocity targets." % self._name)
            return
        return self.move_to_joint_pos_with_vel(target_position_vel_float32array.data[:9], target_position_vel_float32array.data[9:])

    def move_to_joint_pos_with_vel(self, desired_position, desired_velocity):
        '''
        Move Sawyer arm to a target position (interpolate) at a given velocity
        @param target_position: Vector of 9 floats, indicating joint positions for the 7 DoF arm and positions for the 2 DoF gripper's finger positions
        @param target_velocities: Vector of 9 floats, indicating the speed that each joint should move to reach its position.
        '''

        target_position = list(desired_position)
        target_velocity = list(desired_velocity)

        joints_list = self._arm_dof_indices + self._gripper_dof_indices
        if len(target_velocity) != len(target_position):
            rospy.logwarn("Different sizes of target positions (%d) and velocities (%d) passed to move_to_joint_pos_with_vel!" % (len(target_position), len(target_velocity)))
            return

        if len(target_position) != 7 and len(target_position) != 9:
            rospy.logwarn("Invalid joint configuration/velocities provided for Sawyer %s. Function requires lists to be of length 1-7 floats (arm DoF only) or 9 floats (7 arm + 2 gripper DoF)" % self._name)
            return 

        if len(target_position) == 7:
            gripper_pos = p.getJointStates(self._simulator_id, self._gripper_dof_indices)
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
        #p.setJointMotorControlArray(self._simulator_id, joints_list[:len(target_positions)], p.POSITION_CONTROL, targetPositions=target_position, targetVelocities=target_velocity)

        for i,j_idx in enumerate(joints_list):
            p.setJointMotorControl2(self._simulator_id, j_idx, p.POSITION_CONTROL, target_position[i], target_velocity[i], maxVelocity=target_velocity[i])


    def get_current_joint_states(self):
        position = []
        joint_states = p.getJointStates(self._simulator_id, self._arm_dof_indices + self._gripper_dof_indices)
        for entry in joint_states:
            position.append(entry[0])
        return position

    def execute_trajectory_callback(self, trajectory_json_string):
        if self._executing_trajectory: rospy.logwarn("Current trajectory for %s not finished executing, but new trajectory received!" % self._name)
        traj_data = json.loads(trajectory_json_string.data)
        self.execute_trajectory(traj_data)

    def execute_trajectory(self, trajectory_data):
        '''
        Execute a trajectory with the Sawyer arm given positions and timings. This function computes the velocities needed to make the timeline.
        Ex: trajectory_data = [(1., [0,0,0,0,0,0,0]), (2.5, [1,0,0,0,0,0,0,0]), (4, [1,0,2.9,1.23,1.52,0,0,0])]
            Sends robot to 3 waypoints over the course of 4 seconds
        @param trajectory_data Vector of (time, joint configuration) tuples, indicating which joint positions the robot should achieve at which times. Set time=0 for each waypoint if you don't care about timing. Joint configuration vectors can be 7, 8, or 9 floats corresponding to the parameter for move_to_joint_pos (7: arm only, 8: arm + gripper %open, 9: arm + gripper finger positions)
        '''

        joint_positions = []
        joint_velocities = []

        # Initial robot position
        last_time = 0
        last_position = self.get_current_joint_states()

        joint_positions = [copy.copy(last_position)]
        joint_velocities = [[0]*9]

        for waypoint in trajectory_data:
            target_duration = waypoint[0] - last_time
            target_position = waypoint[1]
            target_velocities = []

            if target_duration < 0.001:                
                target_duration = 0.001 # No duration given, avoid divide-by-zero and move quickly

            # Compute distance from current position, compute per-joint velocity to reach in (t - t_{-1}) seconds
            # Each waypoint should have 7-9 values
            if len(target_position) < 7 or len(target_position) > 9:
                rospy.logwarn("Bad trajectory waypoint passed to Sawyer %s. Had length %d. Aborting trajectory." % (self._name, len(target_position)))
                return

            # target_position will be 9-DoF vector for arm+gripper position after this code block
            if len(target_position) == 7:
                target_position = target_position[:7] + last_position[7:9] # Keep old gripper position
            elif len(target_position) == 8: # Arm + Gripper %
                next_pos_gripper = self.get_gripper_pct_finger_positions(target_position[7])
                target_position = target_position[:7] + next_pos_gripper

            # Arm + Gripper velocity
            max_velocities = self._arm_joint_velocity_max + self._gripper_joint_velocity_max
            for i in range(len(target_position)):
                distance_to_cover = abs(target_position[i] - last_position[i])
                velocity = min(distance_to_cover / target_duration, max_velocities[i])
                target_velocities.append(velocity)
            
            joint_positions.append(target_position)
            joint_velocities.append(target_velocities)

            last_time = waypoint[0]
            last_position = target_position # 9-DoF arm+gripper position vector

        # Now that joint_positions and joint_velocities are populated, we can execute the trajectory
        sim = Simulator.get_instance()
        sim.set_robot_trajectory(self._simulator_id, joint_positions, joint_velocities)
            
    def check_if_at_position(self, pos, epsilon=0.2):
        '''
        Returns True if the robot's joints are within (epsilon) of pos, false otherwise
        @param pos Vector of length 7, 8, or 9, corresponding to arm position, arm+gripper%, or arm+gripper position
        '''
        if len(pos) < 7 or len(pos) > 9: 
            rospy.logwarn("Invalid position given to check_if_at_position. Must be length 7, 8, or 9 for Sawyer.")
            return False

        cur_pos = self.get_current_joint_states()

        if len(pos) == 7:
            cur_pos = cur_pos[:7]
        elif len(pos) == 8:
            pos = pos[:7] + self.get_gripper_pct_finger_positions(pos[7])

        dist = np.linalg.norm(np.array(pos) - np.array(cur_pos))

        #rospy.loginfo("Checking if Sawyer is at %s. (dist=%g) %s" % (str(pos), dist, str(dist <= epsilon)))
        if dist <= epsilon: return True
        return False

    def get_gripper_pct_finger_positions(self, pct_gripper_open):
        '''
        Returns the target position of each gripper finger given a percentage of how open the gripper should be
        @param pct_gripper_open Value in range [0.,1.] describing how open the gripper should be
        '''
        pct_gripper_open = max(0.,min(1.,pct_gripper_open))
        max_displacement = 0
        for limit in self._gripper_joint_limits:
            max_displacement += limit[1] - limit[0]
        
        total_displacement = pct_gripper_open * max_displacement
        left_position = total_displacement/2.
        right_position = total_displacement/-2.

        return left_position, right_position
