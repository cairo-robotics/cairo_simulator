"""Summary
"""
import pybullet as p

from cairo_simulator.link import get_link_pairs


class DisabledCollisionsContext(): 

    """Summary
    
    Attributes:
        simulator (TYPE): Description
    """
    
    def __init__(self, simulator): 
        """Summary
        
        Args:
            simulator (TYPE): Description
        """
        self.simulator = simulator
          
    def __enter__(self): 
        """Summary
        """
        self._disable_robot_collisions()
        self._disable_simobj_collisions()
      
    def __exit__(self, exc_type, exc_value, exc_traceback): 
        """Summary
        
        Args:
            exc_type (TYPE): Description
            exc_value (TYPE): Description
            exc_traceback (TYPE): Description
        """
        self._enable_robot_collisions()
        self._enable_simobj_collisions()

    def _disable_robot_collisions(self):
        """Summary
        """
        for robot in self.simulator._robots.keys():
            links = [idx for idx in range(-1, p.getNumJoints(robot))]
            for link in links:
                p.setCollisionFilterGroupMask(robot, link, 0, 0)

    def _disable_simobj_collisions(self):
        """Summary
        """
        for sim_obj in self.simulator._objects.keys():
            p.setCollisionFilterGroupMask(sim_obj, 0, 0, 0)

    def _enable_robot_collisions(self):
        """Summary
        """
        for robot in self.simulator._robots.keys():
            links = [idx for idx in range(-1, p.getNumJoints(robot))]
            for link in links:
                p.setCollisionFilterGroupMask(robot, link, 0, 1)

    def _enable_simobj_collisions(self):
        """Summary
        """
        for sim_obj in self.simulator._objects.keys():
            p.setCollisionFilterGroupMask(sim_obj, 0, 0, 1)



def link_collision(body1, link1, body2, link2, max_distance=0):
    """Summary
    
    Args:
        body1 (TYPE): Description
        link1 (TYPE): Description
        body2 (TYPE): Description
        link2 (TYPE): Description
        max_distance (int, optional): Description
    
    Returns:
        TYPE: Description
    """
    return len(p.getClosestPoints(bodyA=body1, bodyB=body2, distance=max_distance,
                              linkIndexA=link1, linkIndexB=link2,
                              physicsClientId=0)) > 0


def self_collision_test(joint_configuration, robot, excluded_pairs):
    """Summary
    
    Args:
        joint_configuration (TYPE): Description
        robot (TYPE): Description
        excluded_pairs (TYPE): Description
    
    Returns:
        TYPE: Description
    """
    robot_id = robot.get_simulator_id()
    check_link_pairs = get_link_pairs(robot_id, excluded_pairs=excluded_pairs)

    # Store current joint state information
    cur_pos = p.getJointStates(robot._simulator_id, robot._arm_dof_indices)

    # Set new configuration and get link states
    for i, idx in enumerate(robot._arm_dof_indices):
        p.resetJointState(robot._simulator_id, idx, targetValue=joint_configuration[i], targetVelocity=0, physicsClientId=0)

    self_collisions = []
    for link1, link2 in check_link_pairs:
        if link1 != link2:
            if link_collision(body1=robot_id, body2=robot_id, max_distance=0,
                          link1=link1, link2=link2):
                self_collisions.append(True)

    # Restore previous joint states
    for joint, idx in enumerate(robot._arm_dof_indices):
        p.resetJointState(robot._simulator_id, idx, targetValue=cur_pos[joint][0], targetVelocity=cur_pos[joint][1])

    if any(self_collisions):
        return False
    else:
        return True