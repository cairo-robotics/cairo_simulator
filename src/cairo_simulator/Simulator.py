import time
import json
import os
import sys
from abc import ABC, abstractmethod
import numpy as np
import rospy
from std_msgs.msg import Float32MultiArray
from std_msgs.msg import String, Empty
from geometry_msgs.msg import PoseStamped
import pybullet as p
import pybullet_data


ASSETS_PATH = os.path.dirname(os.path.abspath(__file__)) + '/../../assets/' # Find ./cairo_simulator/assets/ from ./cairo_simulator/src/cairo_simulator/

class Simulator:
    __instance = None    

    @staticmethod
    def get_instance():
        if Simulator.__instance is None:
            from cairo_simulator.Manipulators import Manipulator
            Simulator()
        return Simulator.__instance

    def __init__(self, use_real_time=True):
        if Simulator.__instance is not None:
            raise Exception("You may only initialize -one- simulator per program! Use get_instance instead.")
        else:
            Simulator.__instance = self

        self.__init_bullet()
        self.__init_vars(use_real_time)
        self.__init_ros()

    def __del__(self):
        p.disconnect()

    def __init_vars(self, use_real_time):
        self._estop = False # Global e-stop
        self._trajectory_queue = {} # Dict mapping robot ids to (joint_positions, joint_velocities) tuple
        self._trajectory_queue_timers = {} # Dict mapping robot ids to the time that their trajectory execution timeout clock started.
        self._robots = {} # Dict mapping robot ids to Robot objects
        self._objects = {} # Dict mapping object ids to SimObject objects
        self._motion_timeout = 10. # 10s timeout from robot motion
        self._use_real_time = use_real_time
        self._state_broadcast_rate = .1 # 10Hz Robot + Object state broadcast
        self.__last_broadcast_time = 0 # Keep track of last time states were broadcast
        self.__sim_time = 0 # Used if not using real-time simulation. Increments at time_step
        self._sim_timestep = 1./240. # If not using real-time mode, amount of time to pass per step() call
        self.set_real_time(use_real_time)

    def __init_bullet(self):
        # Simulation world setup
        self._physics_client = p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0,0,-9.81)
        id_plane = p.loadURDF("plane.urdf")

    def __init_ros(self):
        rospy.Subscriber("/sim/estop_set", Empty, self.estop_set_callback)
        rospy.Subscriber("/sim/estop_release", Empty, self.estop_release_callback)

    def set_real_time(self, val):
        if val is False:
            p.setRealTimeSimulation(0)
            self._use_real_time = False
        elif val is True:
            p.setRealTimeSimulation(1)
            self._use_real_time = True
        else:
            rospy.logerr("Invalid realtime value given to Simulator.set_real_time: Expected True or False.")

    def step(self):
        if self._use_real_time is False:
            p.stepSimulation()
            self.__sim_time += self._sim_timestep
        else:
            self.__sim_time = time.time()

        cur_time = self.__sim_time
 
        if cur_time - self.__last_broadcast_time > self._state_broadcast_rate:
            self.publish_robot_states()
            self.publish_object_states()
            self.__last_broadcast_time = cur_time
        
        self.process_trajectory_queues()

    def estop_set_callback(self, data):
        self._estop = True
        for id in g_trajectory_queue.keys():
            clear_trajectory_queue(id)

    def estop_release_callback(self, data):
        self._estop = False

    def publish_robot_states(self):
        for id in self._robots.keys():
            self._robots[id].publish_state()
    
    def publish_object_states(self):
        for id in self._objects.keys():
            self._objects[id].publish_state()

    def process_trajectory_queues(self):
        cur_time = time.time()

        for id in self._trajectory_queue.keys():
            if self._trajectory_queue[id] is None: continue # Nothing on queue

            # Check if robot is at the first pos vector off the robot's pos/vel tuple
            pos_vector, vel_vector = self._trajectory_queue[id]
            next_pos = pos_vector[0] # First entry of trajectory's position vector
            assigned_time = self._trajectory_queue_timers[id] # Time last commanded
            if assigned_time is None or self._robots[id].check_if_at_position(next_pos) is True:
                # Robot is at this position, get the next position and velocity targets and remove from trajectory_queue

                # Check if trajectory is finished
                if len(pos_vector) == 1: # Reached the last position in the trajectory
                    self.clear_trajectory_queue(id)
                    continue

                next_vel = vel_vector[0] # First entry of the trajectory's joint velocity vector
                # Send command to the robot controller
                self._robots[id]._executing_trajectory = True

                if isinstance(self._robots[id], Manipulator):
                    self._robots[id].move_to_joint_pos_with_vel(next_pos, next_vel)
                else:
                    rospy.logerr("No mechanism for handling trajectory execution for Robot Type %s" % (str(type(self._robots[id]))))
                    continue

                # Update trajectory_queue_timer
                self._trajectory_queue_timers[id] = cur_time
            elif cur_time - assigned_time > self._motion_timeout:
                # Action timed out, abort trajectory
                rospy.logwarn("Trajectory for robot %d timed out! Aborting remainder of trajectory." % id)
                self.clear_trajectory_queue(id)
                continue

    def add_object(self, simobj_obj):
        id = simobj_obj.get_simulator_id()
        self._objects[id] = simobj_obj
    
    def remove_object(self, simobj_id):
        if simobj_id in self._objects:
            del self._objects[simobj_id]
        else:
            rospy.logerr("Tried to delete object %d, which Simulator doesn't think exists" % simobj_id)

    def add_robot(self, robot_obj):        
        id = robot_obj.get_simulator_id()
        self._robots[id] = robot_obj
        self.add_robot_to_trajectory_queue(id)

    def add_robot_to_trajectory_queue(self, id):        
        self._trajectory_queue[id] = None
        self._trajectory_queue_timers[id] = None

    def clear_trajectory_queue(self, id):
        if id not in self._trajectory_queue.keys(): return
        self._trajectory_queue[id] = None
        self._trajectory_queue_timers[id] = None
        self._robots[id]._executing_trajectory = False

    def set_robot_trajectory(self, id, joint_positions, joint_velocities):        
        self._trajectory_queue[id] = (copy.copy(joint_positions), copy.copy(joint_velocities))
        self._trajectory_queue_timers[id] = None

    def load_scene_file(self, sdf_file, obj_name_prefix):
        sim_id_array = p.loadSDF(sdf_file)
        for i, id in enum(sim_id_array):
            obj = SimObject(obj_name_prefix + str(i), id)



class SimObject():
    def __init__(self, object_name, model_file_or_sim_id, position=(0,0,0), orientation=(0,0,0,1)):
        self._name = object_name

        if isinstance(model_file_or_sim_id, int):
            self._simulator_id = model_file_or_sim_id
        else:
            self._simulator_id = self._load_model_file_into_sim(model_file_or_sim_id)
            self.move_to_pose(position, orientation)

        if self._simulator_id is None:
            rospy.logerr("Couldn't load object model from %s" % model_file)
            return None

        Simulator.get_instance().add_object(self)

        self._state_pub = rospy.Publisher("/%s/pose" % self._name, PoseStamped, queue_size=1)

    def publish_state(self):
        pose = PoseStamped()   
        pos, ori = self.get_pose()
        pose.pose.position.x, pose.pose.position.y, pose.pose.position.z = pos
        pose.pose.orientation.x, pose.pose.orientation.y, pose.pose.orientation.z, pose.pose.orientation.w = ori
        self._state_pub.publish(pose)

    def get_simulator_id(self):
        return self._simulator_id

    def _load_model_file_into_sim(self, filepath):
        sim_id = None
        if filepath[-5:] == '.urdf':
            sim_id = p.loadURDF(filepath, flags=p.URDF_MERGE_FIXED_LINKS)
        elif filepath[-4:] == '.sdf':
            sim_id = p.loadSDF(filepath)
            if isinstance(sim_id, tuple): sim_id = sim_id[0]
        return sim_id

    def get_pose(self):
        '''
        Returns position and orientation vector, e.g.,: ((x,y,z), (x,y,z,w))
        '''
        return p.getBasePositionAndOrientation(self._simulator_id)    

    def move_to_pose(self, position_vec=None, orientation_vec=None):
        '''
        Sets an object's position and orientation in the simulation.
        @param position_vec (Optional) Desired [x,y,z] position of the object. Current position otherwise.
        @param orientation_vec (Optional) Desired orientation of the object as either quaternion or euler angles. Current angle otherwise.
        '''
        cur_pos, cur_ori = p.getBasePositionAndOrientation(self._simulator_id)
        if orientation_vec is None: orientation_vec = cur_ori
        if position_vec is None: position_vec = cur_pos
        if len(orientation_vec) == 3: orientation_vec = p.getQuaternionFromEuler(orientation_vec)
        p.resetBasePositionAndOrientation(self._simulator_id, position_vec, orientation_vec)


class Robot(ABC):
    '''
        Abstract Base Class for a Robot in PyBullet


        If adding a new robot, it can be helpful to check the joint info loaded from the URDF:
            import pdb
            for i in range(p.getNumJoints(self._simulator_id)):
                print("Joint %d: %s" % (i, str(p.getJointInfo(self._simulator_id,i))))
            pdb.set_trace()
    '''
    def __init__(self, robot_name, urdf_file, x, y, z, urdf_flags=p.URDF_MERGE_FIXED_LINKS):   
        """
        Initialize a Robot at coordinates (x,y,z) and add it to the simulator manager

        Warning: Including p.URDF_USE_SELF_COLLISION is buggy right now due to URDF issues and is not recommended
        """ 
        super().__init__()
        self._name = robot_name
        self._executing_trajectory = False
        self._state = None

        self._pub_robot_state = rospy.Publisher('/%s/robot_state'%self._name, Float32MultiArray, queue_size=0)
        self._simulator_id = p.loadURDF(urdf_file, [x,y,z], flags=urdf_flags)

        # Register with Simulator manager
        sim = Simulator.get_instance()
        sim.add_robot(self)

    def _populate_dof_indices(self, dof_name_list):
        '''
        Given a list of DoF names (e.g.: ['j0', 'j1', 'j2']) find their corresponding joint indices for use with p.getJointInfo to retrieve state.

        @param sim_id ID of the entity to find DoF indices for
        @param dof_name_list List of joint names in the order desired
        @returns List of indices into p.getJointInfo corresponding to the dof_names requested
        '''
        dof_index_list = []
        for dof_name in dof_name_list:
            found = False
            for i in range(p.getNumJoints(self._simulator_id)):
                joint_info = p.getJointInfo(self._simulator_id,i)
                if joint_info[1].decode('UTF-8') == dof_name:
                    dof_index_list.append(i)
                    found = True
                    break
            if found is False: 
                rospy.logerr("Could not find joint %s in robot id %d" % (dof_name, self._simulator_id))
                return []
        
        return dof_index_list
            


    def get_simulator_id(self):
        return self._simulator_id

    @abstractmethod
    def publish_state(self):
        '''
        Publish robot state onto a ROS Topic
        '''
        pass
