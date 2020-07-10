"""
Classes and methods to support collision functionality withing PyBullet, and to inject into 
interfaces for motion planning.
"""
import pybullet as p

__all__ = ['DisabledCollisionsContext', 'link_collision', 'self_collision_test']

class DisabledCollisionsContext(): 

    """
    Python Context Manager that disables collisions. Includes all self collisions, and collisions between SimObjects and Robots.
    
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
        for sim_obj in self.simulator._objects.keys():
            if sim_obj not in self.excluded_bodies:
                p.setCollisionFilterGroupMask(sim_obj, 0, 0, 0)

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
        for sim_obj in self.simulator._objects.keys():
            p.setCollisionFilterGroupMask(sim_obj, 0, 0, 1)



def link_collision(body1, link1, body2, link2, max_distance=0):
    """
    Test for the collision between link1 of body1 and link2 of body2. This method relies on p.GetClosestPoints.
    Using a max_distinace of 0 will test for exact collision given overlap of bounding boxes.
    
    Args:
        body1 (int): PyBullet body ID.
        link1 (int): Link/joint index.
        body2 (int): PyBullet body ID.
        link2 (int): Link/joint index.
        max_distance (int, optional): Range within which to test for collision.
    
    Returns:
        bool: True if in collision, else False.
    """
    return len(p.getClosestPoints(bodyA=body1, bodyB=body2, distance=max_distance,
                              linkIndexA=link1, linkIndexB=link2,
                              physicsClientId=0)) > 0


def self_collision_test(joint_configuration, robot, link_pairs):
    """
    Tests whether a give joint configuration will result in self collision. 

    It sets the robot state to the test configuration. Every link pair of the robot is checked for collision.
    If every link pair is collision free, the function returns true, else if there is a single collision, the function returns false.
    
    Args:
        joint_configuration (list): The joint configuration to test for self-collision
        robot (int): PyBullet body ID.
    
    Returns:
        bool: True if no self-collision, else False.
    """
    robot_id = robot.get_simulator_id()

    # Set new configuration and get link states
    for i, idx in enumerate(robot._arm_dof_indices):
        p.resetJointState(robot._simulator_id, idx, targetValue=joint_configuration[i], targetVelocity=0, physicsClientId=0)

    self_collisions = []
    for link1, link2 in link_pairs:
        if link1 != link2:
            if link_collision(body1=robot_id, body2=robot_id, max_distance=0,
                          link1=link1, link2=link2):
                self_collisions.append(True)

    if any(self_collisions):
        return False
    else:
        return True