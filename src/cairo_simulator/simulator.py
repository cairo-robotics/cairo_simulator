"""
Core classes / abstract classes that interface with PyBullet.
"""
import os
import sys
import time
from abc import ABC, abstractmethod

if os.environ.get('ROS_DISTRO'):
    import rospy
    from std_msgs.msg import Float32MultiArray
    from std_msgs.msg import Empty
    from geometry_msgs.msg import PoseStamped

import pybullet as p
import pybullet_data

from cairo_simulator.log import Logger


def rosmethod(func):
    def _decorator(self, *args, **kwargs):
        if not Simulator.is_instantiated() or not Simulator.using_ros():
            raise Exception("Cannot use ROS based method without an instantiated Simulator set to use ROS.")
        return func(self, *args, **kwargs)
    return _decorator


class Simulator:

    """
    Singleton encapsulating the PyBullet simulator.
    """

    __instance = None

    @staticmethod
    def get_instance():
        """
        Get the simulator instance.

        Returns:
            TYPE: Instance of class Simulator

        Raises:
            Exception: If no Simulator has been instantiated, this method is not valid.

        """
        if Simulator.__instance is None:
            raise Exception(
                "You must initialize -one- simulator per program! Use of get_instance can only occur after instantiation.")
        return Simulator.__instance

    @staticmethod
    def is_instantiated():
        """
        Checks if the singleton is instantiated.

        Returns:
            bool: True, if instantiated, else False.
        """
        if Simulator.__instance is not None:
            return True
        else:
            return False

    @staticmethod
    def using_ros():
        """
        Checks if Simulator environment will use ROS integrations.

        Returns:
            bool: True, if using ROS, else False.
        """
        if Simulator.__instance.use_ros:
            return True
        else:
            return False

    @staticmethod
    def get_logger():
        return Simulator.__instance.logger

    def __init__(self, logger=None, use_real_time=True, use_gui=True, use_ros=False):
        """
        Args:
            use_real_time (bool, optional): Whether or not to use real_time for simulation steps.
            gui (bool, optional): Whether or not to display / render the simulator GUI

        Raises:
            Exception: One may only construct the simulator once and instead must ue get_instnct method of already instantiated. 
        """
        if Simulator.__instance is not None:
            raise Exception(
                "You may only initialize -one- simulator per program! Use get_instance instead.")
        else:
            Simulator.__instance = self

        self.logger = logger if logger is not None else Logger(handlers=['logging'])
        self.__init_bullet(gui=use_gui)
        self.__init_vars(use_real_time)
        self.use_ros = use_ros
        if self.use_ros:
            if 'rospy' not in sys.modules:
                raise  'ROS shell environment has not been sourced as rospy could not be imported.'
            self.__init_ros()
            self.logger = logger if logger is not None else Logger(handlers=['ros'])

    def __del__(self):
        p.disconnect()

    def __init_vars(self, use_real_time):
        """
        Initializes the Simulator environment and flags e.g. using real time for simulations, time step length, queues, e_stop etc,.

        Args:
            use_real_time (bool): Whether or not to use real_time for simulation steps.
        """
        self._estop = False  # Global e-stop
        # Dict mapping robot ids to (joint_positions, joint_velocities) tuple
        self._trajectory_queue = {}
        # Dict mapping robot ids to the time that their trajectory execution timeout clock started.
        self._trajectory_queue_timers = {}
        self._robots = {}  # Dict mapping robot ids to Robot objects
        self._objects = {}  # Dict mapping object ids to SimObject objects
        self._motion_timeout = 10.  # 10s timeout from robot motion
        self._use_real_time = use_real_time
        self._state_broadcast_rate = .1  # 10Hz Robot + Object state broadcast
        self.__last_broadcast_time = 0  # Keep track of last time states were broadcast
        self.__sim_time = 0  # Used if not using real-time simulation. Increments at time_step
        self._sim_timestep = 1. / \
            240.  # If not using real-time mode, amount of time to pass per step() call
        self.set_real_time(use_real_time)

    def __init_bullet(self, gui=True):
        """
        Initializes PyBullet physics server. E.g. sets gravity, loads a plane. 

        Args:
            gui (bool, optional): Whether or not to utilize the GUI/renderer or connect using the p.DIRECT flag.
        """
        # Simulation world setup
        if gui:
            self._physics_client = p.connect(p.GUI)
        else:
            self._physics_client = p.connect(p.DIRECT)
        # artifact from PyBullet port. Get's the default Bullet assets.
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        # Potential TODO: Remove ground plane out of init. Speed up rendering: we could remove the plane.

    def __init_ros(self):
        """
        Initializes ROS subscribers 
        """
        rospy.Subscriber("/sim/estop_set", Empty, self.estop_set_callback)
        rospy.Subscriber("/sim/estop_release", Empty,
                         self.estop_release_callback)

    def set_real_time(self, val):
        """
        Whether or not to use real time simulation for steps or the set amount of time designated in __init_vars().

        Args:
            val (bool): Boolean flag.
        """
        if val is False:
            p.setRealTimeSimulation(0)
            self._use_real_time = False
        elif val is True:
            p.setRealTimeSimulation(1)
            self._use_real_time = True
        else:
           self.logger.err(
                "Invalid realtime value given to Simulator.set_real_time: Expected True or False.")

    def step(self):
        """
        Steps the simulation forward one time step by wrapping the p.stepSimulation() function call. 
        Also broadcases robot and object states and processes trajectory queues.

        """
        if self._use_real_time is False:
            p.stepSimulation()
            self.__sim_time += self._sim_timestep
        else:
            self.__sim_time = time.time()

        cur_time = self.__sim_time

        if self.use_ros:
            if cur_time - self.__last_broadcast_time > self._state_broadcast_rate:
                self.publish_robot_states()
                self.publish_object_states()
                self.__last_broadcast_time = cur_time

        self.process_trajectory_queues()

    def estop_set_callback(self, data):
        """
        Callback that triggers the Simulator emergency stop and clear trajectory queues.

        Args:
            data (rospy message): Data message but unused.
        """
        self._estop = True
        for traj_id in self._trajectory_queue.keys():
            self.clear_trajectory_queue(traj_id)

    def estop_release_callback(self, data):
        """
        Callback that releases the Simulator's emergency stop.

        Args:
            data (rospy message): Data message but unused.
        """
        self._estop = False

    @rosmethod
    def publish_robot_states(self):
        """
        For each registered Robot, publishes its state.
        """
        for robot_id in self._robots.keys():
            self._robots[robot_id].publish_state()

    @rosmethod
    def publish_object_states(self):
        """
        For each registered SimObject, publishes its state.
        """
        for object_id in self._objects.keys():
            self._objects[object_id].publish_state()

    def process_trajectory_queues(self):
        """
        Executes one step of each trajectory queue.
        """
        cur_time = self.__sim_time
        for traj_id in self._trajectory_queue.keys():
            if self._trajectory_queue[traj_id] is None:
                continue  # Nothing on queue

            if self._trajectory_queue_timers[traj_id] is not None and cur_time - self._trajectory_queue_timers[traj_id] > self._motion_timeout:
                # Action timed out, abort trajectory
                if self.use_ros:
                    self.logger.warn(
                        "Trajectory for robot %d timed out! Aborting remainder of trajectory." % traj_id)
                self.clear_trajectory_queue(traj_id)
                continue

            # Check if robot is at the first position entry off the robot's pos/vel tuple
            # to see if they reached their target waypoint and are ready for the next

            # pos_vector and vel_vector layout: [ original_pos, target_pos, future pos, future pos, ...]
            pos_vector, vel_vector = self._trajectory_queue[traj_id]

            # Check if trajectory is finished
            # Only have original_pos, therefore we reached the last position in the trajectory
            if len(pos_vector) == 1:
                self.clear_trajectory_queue(traj_id)
                continue

            # Original position or last commanded position
            prev_pos = pos_vector[0]
            # First entry of trajectory's position vector
            next_pos = pos_vector[1]
            # First entry of the trajectory's joint velocity vector
            next_vel = vel_vector[1]

            at_pos = self._robots[traj_id].check_if_at_position(prev_pos)
            if (self._trajectory_queue_timers[traj_id] is None) or (at_pos is True):
                # Robot is at this position, get the next position and velocity targets and remove from trajectory_queue
                self._trajectory_queue_timers[traj_id] = cur_time

                attr = getattr(self._robots[traj_id],
                               'move_to_joint_pos_with_vel', None)
                if attr is not None:
                    self._robots[traj_id].move_to_joint_pos_with_vel(
                        next_pos, next_vel)
                    # Increment progress in the trajectory
                    self._trajectory_queue[traj_id][0] = self._trajectory_queue[traj_id][0][1:]
                    # Increment progress in the trajectory
                    self._trajectory_queue[traj_id][1] = self._trajectory_queue[traj_id][1][1:]
                else:
                    if self.use_ros:
                        self.logger.err("No mechanism for handling trajectory execution for Robot Type %s" % (
                            str(type(self._robots[id]))))
                    self.clear_trajectory_queue(traj_id)
                    continue

                # Update trajectory_queue_timer
            elif cur_time - self._trajectory_queue_timers[traj_id] > self._motion_timeout:
                # Action timed out, abort trajectory
                if self.use_ros:
                    self.logger.warn(
                        "Trajectory for robot %d timed out! Aborting remainder of trajectory." % traj_id)
                self.clear_trajectory_queue(traj_id)
                continue

    def add_object(self, simobj_obj):
        """
        Adds a SimObject to the Simulator via a dictionary whos keys are the PyBullet body ID.

        Args:
            simobj_obj (int): PyBullet body ID.
        """
        sim_id = simobj_obj.get_simulator_id()
        self._objects[sim_id] = simobj_obj

    def remove_object(self, simobj_id):
        """
        Removes a SimObject to the Simulator via a dictionary whos keys are the PyBullet body ID.

        Args:
            simobj_id (int): PyBullet body ID.
        """
        if simobj_id in self._objects:
            del self._objects[simobj_id]
        else:
            self.logger.err(
                "Tried to delete object %d, which Simulator doesn't think exists" % simobj_id)

    def add_robot(self, robot_obj):
        """
        Adds a Robot to the Simulator via a dictionary whos keys are the PyBullet body ID.

        Args:
            robot_obj (int): PyBullet body ID.
        """
        sim_id = robot_obj.get_simulator_id()
        self._robots[sim_id] = robot_obj
        self.add_robot_to_trajectory_queue(sim_id)

    def add_robot_to_trajectory_queue(self, robot_id):
        """
        Adds a Robot ID to the trajectory queue and associated timer queue as a key.

        Args:
            robot_obj (int): PyBullet body ID for the robot.
        """
        self._trajectory_queue[robot_id] = None
        self._trajectory_queue_timers[robot_id] = None

    def clear_trajectory_queue(self, id_):
        """
        Clears a trajectory queue for the given id.

        Args:
            id_ (int): PyBullet body ID.
        """
        if id_ not in self._trajectory_queue.keys():
            return
        self._trajectory_queue[id_] = None
        self._trajectory_queue_timers[id_] = None

    def set_robot_trajectory(self, id_, joint_positions, joint_velocities):
        """
        Sets the robot trajectory given as a list of joint positions (configurations) and joint velocities.
        These added as a list of lists to the trajectory queue for the given PyBullet body ID.

        Args:
            id_ (int): PyBullet body ID
            joint_positions (list): List of joint configurations/positions.
            joint_velocities (list): List of associated joint velocities matched to the joint positions.
        """
        self._trajectory_queue[id_] = [
            list(joint_positions), list(joint_velocities)]
        self._trajectory_queue_timers[id_] = None

    def load_scene_file(self, sdf_file, obj_name_prefix):
        """
        Loards a scene file into the PyBullet environment via the p.loadSDF functoin call.

        Args:
            sdf_file (file object): The .sdf file
            obj_name_prefix (str): Object name prefix for objects defined in the scene.
        """
        sim_id_array = p.loadSDF(sdf_file)
        for i, id_ in enumerate(sim_id_array):
            obj = SimObject(obj_name_prefix + str(i), id_)


class SimObject():

    """
    A Simulation Object within the PyBullet physics simulation environment.
    """

    def __init__(self, object_name, model_file_or_sim_id, position=(0,0,0), orientation=(0,0,0,1), fixed_base=0):
        """
        Args:
            object_name (str): Name of the sim object.
            model_file_or_sim_id (file object, int): Either File object or a PyBullet body ID.
            position (tuple, optional): Initial position; 3-tuple
            orientation (tuple, optional): Initial orientation; 4-tuple if quaternion, 3-tuple if Euler angles.

        Raises:
            Exception: Raises an Exception if either Simulator not instantiated or loading from file/if fails.
        """
        self._name = object_name
        if Simulator.is_instantiated():
            self.logger = Simulator.get_logger()
            if isinstance(model_file_or_sim_id, int):
                self._simulator_id = model_file_or_sim_id
                self.logger.info()
            else:
                self._simulator_id = self._load_model_file_into_sim(
                    model_file_or_sim_id, fixed_base)
                self.move_to_pose(position, orientation)

            if self._simulator_id is None:
                raise Exception("Couldn't load object model from %s" %
                                self.model_file_or_sim_id)
            Simulator.get_instance().add_object(self)

            if Simulator.using_ros():
                self.__init_ros()

        else:
            raise Exception(
                "Simulator must be instantiated before creating a SimObject.")

    def __init_ros(self):
        """
        Initializes ROS pubs and subs 
        """
        self._state_pub = rospy.Publisher(
                    "/%s/pose" % self._name, PoseStamped, queue_size=1)

    @rosmethod
    def publish_state(self):
        """
        Publishes the state of the object if using ROS.
        """
        pose = PoseStamped()
        pos, ori = self.get_pose()
        pose.pose.position.x, pose.pose.position.y, pose.pose.position.z = pos
        pose.pose.orientation.x, pose.pose.orientation.y, pose.pose.orientation.z, pose.pose.orientation.w = ori
        self._state_pub.publish(pose)

    def get_simulator_id(self):
        """
        Get's the PyBullet body ID of the Sim Object.

        Returns:
            TYPE: Description
        """
        return self._simulator_id

    def _load_model_file_into_sim(self, filepath, fixed_base=0):
        """
        Loads the URDF usinga filepath into the PyBullet environment.

        Args:
            filepath (str): Filepath of the URDF file.

        Returns:
            int: The PyBullet body ID of the Sim Object.
        """
        sim_id = None
        if filepath[-5:] == '.urdf':
            sim_id = p.loadURDF(filepath, useFixedBase=fixed_base, flags=p.URDF_MERGE_FIXED_LINKS)
        elif filepath[-4:] == '.sdf':
            sim_id = p.loadSDF(filepath)
            if isinstance(sim_id, tuple): 
                for idx, sdf_object_id in enumerate(sim_id[1:]):
                    loaded_object = SimObject("%s_%d"%(self._name, idx), 
                                    sdf_object_id)
                sim_id = sim_id[0]
        return sim_id

    def get_pose(self):
        '''
        Returns position and orientation vector, e.g.,: ((x,y,z), (x,y,z,w))

        Returns:
            tuple: (position tuple, orientation tuple) i.e. ((x,y,z), (x,y,z,w))
        '''
        return p.getBasePositionAndOrientation(self._simulator_id)

    def move_to_pose(self, position_vec=None, orientation_vec=None):
        '''
        Sets an object's position and orientation in the simulation.

        Args:
            position_vec (None, optional):  Desired [x,y,z] position of the object. Current position otherwise.
            orientation_vec (None, optional): Desired orientation of the object as either quaternion or euler angles. Current angle otherwise.
        '''
        print(self._simulator_id)
        cur_pos, cur_ori = p.getBasePositionAndOrientation(self._simulator_id)
        if orientation_vec is None:
            orientation_vec = cur_ori
        if position_vec is None:
            position_vec = cur_pos
        if len(orientation_vec) == 3:
            orientation_vec = p.getQuaternionFromEuler(orientation_vec)
        p.resetBasePositionAndOrientation(
            self._simulator_id, position_vec, orientation_vec)


class Robot(ABC):
    '''
    Abstract Base Class for a Robot in PyBullet


    If adding a new robot, it can be helpful to check the joint info loaded from the URDF:
        import pdb
        for i in range(p.getNumJoints(self._simulator_id)):
            print("Joint %d: %s" % (i, str(p.getJointInfo(self._simulator_id,i))))
        pdb.set_trace()
    '''

    def __init__(self, robot_name, urdf_file, position, orientation=[0,0,0,1], fixed_base=0, urdf_flags=0):   
        """
        Initialize a Robot at pose=(position, orientation) and add it to the simulator manager.

        Warning: Including p.URDF_USE_SELF_COLLISION is buggy right now due to URDF issues and is not recommended

        Args:
            robot_name (str): Name of the robot
            urdf_file (str): Filepath to urdf file.
            position (list): Point [x,y,z]
            orientation (list): Quaternion [x,y,z,w]
            fixed_base (int): 0 if base can be moved, 1 if fixed in place
            urdf_flags (int): Bitwise flags.
        """
        super().__init__()
        self._name = robot_name
        self._state = None
        if Simulator.is_instantiated():
            self.logger = Simulator.get_logger()
            self._simulator_id = p.loadURDF(urdf_file, basePosition=position, 
                baseOrientation=orientation, useFixedBase=fixed_base, flags=urdf_flags)
            # Register with Simulator manager
            sim = Simulator.get_instance()
            sim.add_robot(self)

            if Simulator.using_ros():
                self.__init_ros()

        else:
            raise Exception(
                "Simulator must be instantiated before creating a SimObject.")

    def __init_ros(self):
        self._pub_robot_state = rospy.Publisher(
            '/%s/robot_state' % self._name, Float32MultiArray, queue_size=0)
       
    def _populate_dof_indices(self, dof_name_list):
        '''
        Given a list of DoF names (e.g.: ['j0', 'j1', 'j2']) find their corresponding joint indices for use with p.getJointInfo to retrieve state.

        Args:
            dof_name_list (list): List of joint names in the order desired

        Returns:
            list:  List of indices into p.getJointInfo corresponding to the dof_names requested
        '''
        dof_index_list = []
        for dof_name in dof_name_list:
            found = False
            for i in range(p.getNumJoints(self._simulator_id)):
                joint_info = p.getJointInfo(self._simulator_id, i)
                if joint_info[1].decode('UTF-8') == dof_name:
                    dof_index_list.append(i)
                    found = True
                    break
            if found is False:
                self.logger.err("Could not find joint %s in robot id %d" %
                             (dof_name, self._simulator_id))
                return []

        return dof_index_list
    
    def set_world_pose(self, position, orientation):
        '''
        Set the world pose and orientation of the robot

        Args:
        position (list): World position in the form [x,y,z]
        orientation (list): Quaternion in the form [x,y,z,w]
        '''
        p.resetBasePositionAndOrientation(self._simulator_id, position, orientation)

    def get_simulator_id(self):
        """
        Retrieves the assigned PyBullet body ID.

        Returns:
            int: PyBullet body ID.
        """
        return self._simulator_id

    @abstractmethod
    def publish_state(self):
        '''
        Publish robot state onto a ROS Topic
        '''
        pass
