"""
Classes and methods to support collision functionality withing PyBullet, and to inject into 
interfaces for motion planning.
"""
import pybullet as p
from collections import namedtuple

__all__ = ['DisabledCollisionsContext', 'get_closest_points', 'self_collision_test', 'robot_body_collision_test', 'multi_collision_test']


CollisionInfo = namedtuple('CollisionInfo',
                           """
                           contactFlag
                           bodyUniqueIdA
                           bodyUniqueIdB
                           linkIndexA
                           linkIndexB
                           positionOnA
                           positionOnB
                           contactNormalOnB
                           contactDistance
                           normalForce
                           lateralFriction1
                           lateralFrictionDir1
                           lateralFriction2
                           lateralFrictionDir2
                           """.split())

class DisabledCollisionsContext(): 

    """
    Python Context Manager that disables collisions. Includes all self collisions, and collisions between SimObjects and Robots. 
    
    This does not disable collision checking but enables the robot to be repositioned into a configuration and then checked for collision without causing any physical simulation effects to occur.
    
    Attributes:
        simulator (simulator.Simulator): The Simulator singleton instance.
    """
    
    def __init__(self, simulator, excluded_bodies=[], excluded_body_link_pairs=[]): 
        """
        
        Args:
        simulator (simulator.Simulator): The Simulator singleton instance.  
        excluded_bodies (list): A list of PyBullet body IDs to excluded, generally reserved for SimObjects with only one link.
        excluded_body_link_pairs (list): A list tuples where the first element is the PyBullet Body ID and the second elemend is the link index. 
        """
        self.simulator = simulator
        self.excluded_bodies = excluded_bodies
        self.excluded_body_link_pairs = excluded_body_link_pairs
          
    def __enter__(self):
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING,0) #makes loading faster
        self._disable_robot_collisions()
        self._disable_simobj_collisions()
        self.state_id = p.saveState()
      
    def __exit__(self, exc_type, exc_value, exc_traceback): 
        self._enable_robot_collisions()
        self._enable_simobj_collisions()
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING,1) #re-enable rendering 
        p.restoreState(self.state_id)

    def _disable_robot_collisions(self):
        """
        Disables all self collisions for all robots.
        """
        for robot in self.simulator._robots.keys():
            if robot not in self.excluded_bodies:
                links = [idx for idx in range(-1, p.getNumJoints(robot))]
                for link in links:
                    if (robot, link) not in self.excluded_body_link_pairs:
                        p.setCollisionFilterGroupMask(robot, link, 0, 0)

    def _disable_simobj_collisions(self):
        """
        Disables all collisions between simobjects and robots.
        """
        for robot in self.simulator._robots.keys():
            for sim_obj in self.simulator._objects.keys():
                if sim_obj not in self.excluded_bodies:
                    p.setCollisionFilterGroupMask(robot, sim_obj, 0, 0)

    def _enable_robot_collisions(self):
        """
        Enables all self collisions for all robots.
        """
        for robot in self.simulator._robots.keys():
            links = [idx for idx in range(-1, p.getNumJoints(robot))]
            for link in links:
                p.setCollisionFilterGroupMask(robot, link, 0, 1)

    def _enable_simobj_collisions(self):
        """
        Enables all collisions between simobjects and robots.
        """
        for robot in self.simulator._robots.keys():
            for sim_obj in self.simulator._objects.keys():
                if sim_obj not in self.excluded_bodies:
                    p.setCollisionFilterGroupMask(robot, sim_obj, 0, 1)


def get_closest_points(client_id, body1, body2, link1=None, link2=None, max_distance=0.):
    """
    Test for the collision between link1 of body1 and link2 of body2. This method relies on p.GetClosestPoints.
    Using a max_distinace of 0 will test for exact collision given overlap of bounding boxes. 
    
    Links are optional and the method will adapt to just using the entire body.
    
    Args:
        body1 (int): PyBullet body ID.
        link1 (int): Link/joint index.
        body2 (int): PyBullet body ID.
        link2 (int): Link/joint index.
        max_distance (int, optional): Range within which to test for collision.
    
    Returns:
        CollisionInfo: Named Tuple of Collision information.
    """
    if (link1 is None) and (link2 is None):
        results = p.getClosestPoints(bodyA=body1, bodyB=body2, distance=max_distance, physicsClientId=client_id)
    elif link2 is None:
        results = p.getClosestPoints(bodyA=body1, bodyB=body2, linkIndexA=link1,
                                     distance=max_distance, physicsClientId=client_id)
    elif link1 is None:
        results = p.getClosestPoints(bodyA=body1, bodyB=body2, linkIndexB=link2,
                                     distance=max_distance, physicsClientId=client_id)
    else:
        results = p.getClosestPoints(bodyA=body1, bodyB=body2, linkIndexA=link1, linkIndexB=link2,
                                     distance=max_distance, physicsClientId=client_id)
    return [CollisionInfo(*info) for info in results]


def self_collision_test(joint_configuration, robot, link_pairs, client_id=0):
    """
    Tests whether a given joint configuration will result in self collision. 

    It sets the robot state to the test configuration. Every link pair of the robot is checked for collision.
    If every link pair is collision free, the function returns true, else if there is a single collision, the function returns false.
    
    Args:
        joint_configuration (list): The joint configuration to test for self-collision
        robot (cairo_simulator Robot): The robot tested for self-collision.
        link_pairs (list): 2D pairs of pybullet body links, in this case the pairs of potential in-self-collision pairs. Assumes the first element is a robot link and second element is another robot link.
    
    
    Returns:
        bool: True if no self-collision, else False.
    """
    robot_id = robot.get_simulator_id()

    # Set new configuration
    for i, idx in enumerate(robot._arm_dof_indices):
        p.resetJointState(robot_id, idx, targetValue=joint_configuration[i], targetVelocity=0, physicsClientId=0)


    self_collisions = []
    for link1, link2 in link_pairs:
        if link1 != link2:
            if len(get_closest_points(client_id=client_id, body1=robot_id, body2=robot_id, max_distance=0,
                          link1=link1, link2=link2)) == 0:
                self_collisions.append(True)
            else:
                self_collisions.append(False)
    if all(self_collisions):
        return True
    else:
        return False


def robot_body_collision_test(joint_configuration, robot, object_body_id, client_id=0, max_distance=0):
    """
    Tests whether a given joint configuration will result in collision with an object. 

    It sets the robot state to the test configuration. There are no links test, we're testing whole body ids.
    
    Args:
        joint_configuration (list): The joint configuration to test for robot-object collision.
        robot (int): PyBullet body ID.
        object_body_id: The other object against which to test collision. 
        client_id (int): the physics server client ID.
    Returns:
        bool: True if no self-collision, else False.
    """
    robot_id = robot.get_simulator_id()
     # Set new configuration and get link states
    for i, idx in enumerate(robot._arm_dof_indices):
        p.resetJointState(robot_id, idx, targetValue=joint_configuration[i], targetVelocity=0, physicsClientId=client_id)
    p.performCollisionDetection()
    if len(get_closest_points(client_id=client_id, body1=robot_id, body2=object_body_id, max_distance=max_distance)) == 0:
        return True
    else:
        return False

def multi_collision_test(joint_configuration, robot_object_collision_fns):
    """Runs a set of collision functions 

    Args:
        robot_object_collision_fns (list of functions): list of functions to run that tests for collisions between a robot and a collision object.

    Returns:
        bool: True if no self-collision, else False.
    """
    for robot_object_collision_fn in robot_object_collision_fns:
        if not robot_object_collision_fn(joint_configuration):
            return False
    return True