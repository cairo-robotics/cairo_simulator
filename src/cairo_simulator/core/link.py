"""
Helper classes and methods to perform body and link based queries and filtering for PyBullet simulation bodies.
"""
from itertools import product, combinations

import pybullet as p

from cairo_simulator.core.utils import JointInfo


def getMotorJointStates(robot):
    joint_states = p.getJointStates(robot, range(p.getNumJoints(robot)))
    joint_infos = [JointInfo(*p.getJointInfo(robot, i))
                   for i in range(p.getNumJoints(robot))]
    nonfixed_joint_info = [
        ji for ji in joint_infos if ji.type != p.JOINT_FIXED]
    nonfixed_joint_states = [joint_states[ji.idx]
                             for ji in nonfixed_joint_info]
    joint_names = [ji.name for ji in nonfixed_joint_info]
    joint_positions = [state[0] for state in nonfixed_joint_states]
    joint_velocities = [state[1] for state in nonfixed_joint_states]
    joint_torques = [state[3] for state in nonfixed_joint_states]
    return joint_names, joint_positions, joint_velocities, joint_torques


def getJointStates(robot):
    joint_states = p.getJointStates(robot, range(p.getNumJoints(robot)))
    joint_positions = [state[0] for state in joint_states]
    joint_velocities = [state[1] for state in joint_states]
    joint_torques = [state[3] for state in joint_states]
    return joint_positions, joint_velocities, joint_torques


def get_joint_info_by_name(body, name):
    """
    Returns a JointInfo namedtuple for the given body and joint/link name.

    Args:
        body (int): PyBullet body ID
        name (str): Name of link

    Returns:
        JointInfo: Namedtuple wrapping p.getJointInfo().
    """
    _link_name_to_index = {p.getBodyInfo(body)[0].decode('UTF-8'): -1, }

    for _id in range(p.getNumJoints(body)):
        _name = p.getJointInfo(body, _id)[12].decode('UTF-8')
        _link_name_to_index[_name] = _id
    link_id = _link_name_to_index.get(name)
    return get_joint_info(body, link_id) if link_id is not None else None


def get_joint_info(body, joint):
    """
    Returns a JointInfo namedtuple for the given body and joint/link

    Args:
        body (int): PyBullet body ID.
        joint (int): Joint/link index.

    Returns:
        JointInfo: Namedtuple wrapping p.getJointInfo().
    """
    return JointInfo(*p.getJointInfo(body, joint))


def check_fixed_link(body, link):
    """
    Checks if for the given body and joint/link, if the joint/link is fixed.

    Args:
        body (int): PyBullet body ID.
        link (int): Joint/link index.

    Returns:
        bool: True if fixed link, else False.
    """
    return get_joint_info(body, link).type == p.JOINT_FIXED


def check_moving_link(body, link):
    """
    Checks if for the given body and joint/link, if the joint/link is moving/movable.

    Args:
        body (int): PyBullet body ID.
        link (int): Joint/link index.

    Returns:
        bool: True if a moving link, else False.
    """
    return not check_fixed_link(body, link)


def check_adjacent_links(body, link1, link2):
    """
    Checks if for the given body two joints/links are adjacent.

    Args:
        body (int): PyBullet body ID.
        link1 (int): Joint/link index.
        link1 (int): Joint/link index.

    Returns:
        bool: True if adjacent, else False.
    """
    link1_parent = get_joint_info(body, link1).parent_idx
    link2_parent = get_joint_info(body, link2).parent_idx
    return (link1_parent == link2) or (link2_parent == link1)


def check_shared_parent_link(body, link1, link2):
    """
    Checks if for the given body two joints/links share the same parent link.

    Args:
        body (int): PyBullet body ID.
        link1 (int): Joint/link index.
        link1 (int): Joint/link index.

    Returns:
        bool: True if shared parent, else False.
    """
    link1_parent = get_joint_info(body, link1).parent_idx
    link2_parent = get_joint_info(body, link2).parent_idx
    return link1_parent == link2_parent


def get_movable_links(body):
    """
    For a given PyBullet body, returns all links that are moving/movable.

    Args:
        body (int): PyBullet body ID.

    Returns:
        list: List of movable link IDs/indices.
    """
    return [link for link in range(0, p.getNumJoints(body)) if check_moving_link(body, link)]


def get_fixed_links(body):
    """
    For a given PyBullet body, returns all links that are fixed.

    Args:
        body (int): PyBullet body ID.

    Returns:
        list: List of fixed link IDs/indices.
    """
    return [link for link in range(0, p.getNumJoints(body)) if check_fixed_link(body, link)]


def filter_equivalent_pairs(pairs):
    """
    Removes all link pairs that have the same ID/index.

    Args:
        pairs (list): List of link pair tuples to filter.

    Returns:
        list: Filtered link pairs.
    """
    return [pair for pair in pairs if pair[0] != pair[1]]


def get_link_pairs(body, excluded_pairs=[]):
    """
    Gets all link pairs for a given body, less the ecluded_pairs set.
    ~ O(N^2)

    Args:
        body (int): The PyBullet body ID.
        excluded_pairs (list, optional): The set of pairs to ignore / eclude with returning all link pairs.

    Returns:
        list: List of link pairs.
    """
    movable_links = get_movable_links(body)
    fixed_links = get_fixed_links(body)
    link_pairs = list(product(movable_links, fixed_links))
    link_pairs.extend(list(combinations(movable_links, 2)))
    link_pairs = [
        pair for pair in link_pairs if not check_adjacent_links(body, *pair)]
    link_pairs = [
        pair for pair in link_pairs if not check_shared_parent_link(body, *pair)]
    link_pairs = [
        pair for pair in link_pairs if pair not in excluded_pairs and pair[::-1] not in excluded_pairs]
    link_pairs = filter_equivalent_pairs(link_pairs)
    return link_pairs


def get_link_from_joint(robot_id):
    """Summary

    Args:
        robot_id (TYPE): Description

    Returns:
        TYPE: Description
    """
    _link_name_to_index = {p.getBodyInfo(robot_id)[0].decode('UTF-8'): -1, }

    for _id in range(p.getNumJoints(robot_id)):
        _name = p.getJointInfo(robot_id, _id)[12].decode('UTF-8')
        _link_name_to_index[_name] = _id
    return _link_name_to_index
