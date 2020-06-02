"""Summary
"""
from itertools import product, combinations

import pybullet as p

from planning.planning_utils import JointInfo


def get_joint_info(body, joint):
    """Summary
    
    Args:
        body (TYPE): Description
        joint (TYPE): Description
    
    Returns:
        TYPE: Description
    """
    return JointInfo(*p.getJointInfo(body, joint))

def check_fixed_link(body, link):
    """Summary
    
    Args:
        body (TYPE): Description
        link (TYPE): Description
    
    Returns:
        TYPE: Description
    """
    return get_joint_info(body, link).type == p.JOINT_FIXED

def check_moving_link(body, link):
    """Summary
    
    Args:
        body (TYPE): Description
        link (TYPE): Description
    
    Returns:
        TYPE: Description
    """
    return not check_fixed_link(body, link)

def check_adjacent_links(body, link1, link2):
    """Summary
    
    Args:
        body (TYPE): Description
        link1 (TYPE): Description
        link2 (TYPE): Description
    
    Returns:
        TYPE: Description
    """
    link1_parent = get_joint_info(body, link1).parent_idx
    link2_parent = get_joint_info(body, link2).parent_idx
    return (link1_parent == link2) or (link2_parent == link1)

def check_shared_parent_link(body, link1, link2):
    """Summary
    
    Args:
        body (TYPE): Description
        link1 (TYPE): Description
        link2 (TYPE): Description
    
    Returns:
        TYPE: Description
    """
    link1_parent = get_joint_info(body, link1).parent_idx
    link2_parent = get_joint_info(body, link2).parent_idx
    return link1_parent == link2_parent

def get_movable_links(body):
    """Summary
    
    Args:
        body (TYPE): Description
    
    Returns:
        TYPE: Description
    """
    return [link for link in range(0, p.getNumJoints(body)) if check_moving_link(body, link)]

def get_fixed_links(body):
    """Summary
    
    Args:
        body (TYPE): Description
    
    Returns:
        TYPE: Description
    """
    return [link for link in range(0, p.getNumJoints(body)) if check_fixed_link(body, link)]

def filter_equivalent_pairs(pairs):
    """Summary
    
    Args:
        pairs (TYPE): Description
    
    Returns:
        TYPE: Description
    """
    return [pair for pair in pairs if pair[0] != pair[1]]

def get_link_pairs(body, excluded_pairs=set()):
    """Summary
    
    Args:
        body (TYPE): Description
        excluded_pairs (TYPE, optional): Description
    
    Returns:
        TYPE: Description
    """
    movable_links = get_movable_links(body)
    fixed_links = get_fixed_links(body)
    link_pairs = list(product(movable_links, fixed_links))
    link_pairs.extend(list(combinations(movable_links, 2)))
    link_pairs = [pair for pair in link_pairs if not check_adjacent_links(body, *pair)]
    link_pairs = [pair for pair in link_pairs if not check_shared_parent_link(body, *pair)]
    link_pairs = [pair for pair in link_pairs if pair not in excluded_pairs and pair[::-1] not in excluded_pairs]
    return link_pairs

def get_link_from_joint(robot_id):
    """Summary
    
    Args:
        robot_id (TYPE): Description
    
    Returns:
        TYPE: Description
    """
    _link_name_to_index = {p.getBodyInfo(robot_id)[0].decode('UTF-8'):-1,}
        
    for _id in range(p.getNumJoints(robot_id)):
        _name = p.getJointInfo(robot_id, _id)[12].decode('UTF-8')
        _link_name_to_index[_name] = _id
    return _link_name_to_index
