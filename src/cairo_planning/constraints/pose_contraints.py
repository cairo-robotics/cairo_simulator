from math import fabs

import numpy as np
from pyquaternion import Quaternion

from cairo_planning.constraints.utils import angle_between, quaternion_to_euler


def orientation(correct_orientation, current_orientation, threshold_angle, axis):
    cone_eval = cone(correct_orientation, current_orientation, threshold_angle, axis)
    twist_eval = twist(correct_orientation, current_orientation, threshold_angle, axis)
    if cone_eval and twist_eval:
        return True
    else:
        return False

def cone(correct_orientation, current_orientation, threshold_angle, axis):
    """
    TODO: Format documentation for consistency.
    Determines if an object's orientation within a cone centered around a given axis and with a threshold angle dead center in that cone. 
    Parameters
    ----------
    correct_orientation : list
        The correct orientation [w, x, y, z]
    current_orientation : list
        Current orientation.
    threshold_angle : float/int
        Threshold angle within which the angle of deviation indicates the pose is correct.
    Returns
    -------
    : int
        1 if within threshold angle, 0 otherwise.
    """

    if axis == "x":
        ref_vec = np.array([1., 0., 0.])  # Unit vector in the +x direction
    elif axis == "y":
        ref_vec = np.array([0., 1., 0.])  # Unit vector in the +y direction
    else:
        ref_vec = np.array([0., 0., 1.])  # Unit vector in the +z direction
    correct_q = Quaternion(correct_orientation[0],
                           correct_orientation[1],
                           correct_orientation[2],
                           correct_orientation[3])

    current_q = Quaternion(current_orientation[0],
                           current_orientation[1],
                           current_orientation[2],
                           current_orientation[3])

    q_prime = correct_q.inverse * current_q

    rot_vec = q_prime.rotate(ref_vec)
    angle = np.rad2deg(angle_between(ref_vec, rot_vec))

    # Classification
    if angle < threshold_angle:
        return 1
    else:
        return 0


def twist(correct_orientation, current_orientation, threshold_angle, axis="z"):
    """
    TODO: Format documentation for consistency.
    Determines whether or not a current_pose twisted about the given axis is within a threshold angle.
    Parameters
    ----------
    correct_orientation : list
        The correct orientation.
    current_pose : list
        Current orientation.
    threshold_angle : float/int
        Threshold angle within which the angle of deviation indicates the pose is correct.
    axis : str
        Axis against which to measure deviation.
    Returns
    -------
    : int
        1 if within threshold angle, 0 otherwise.
    """
    ref_vec = [1.0, 1.0, 1.0]

    correct_q = Quaternion(correct_orientation[0],
                           correct_orientation[1],
                           correct_orientation[2],
                           correct_orientation[3])

    current_q = Quaternion(current_orientation[0],
                           current_orientation[1],
                           current_orientation[2],
                           current_orientation[3])


    q_prime = current_q.inverse * correct_q

    q_prime_fixed = correct_q * q_prime * correct_q.inverse

    roll, _, _ = quaternion_to_euler(q_prime_fixed[1], q_prime_fixed[2], q_prime_fixed[3], q_prime_fixed[0])

    # Classification
    if -threshold_angle <= roll and roll <= threshold_angle:
        return 1
    else:
        return 0


def over_under(above_pose, below_pose, threshold_distance, axis="z"):
    """
    Determines whether one pose is above another pose given a threshold distance of deviation.
    The threshold distance of deviation means the radial distance around the vertical axis that
    determines if the classifier will register as true of false. The above object must have a greater
    positional value for the given axis dimension.
    Parameters
    ----------
    above_pose : geometry_msgs/Pose
        The above pose.
    below_pose : geometry_msgs/Pose
        The below pose.
    threshold_distance : float/int
        Threshold distance within which the poses are close enough to be over/under
    axis : str
        Vertical axis.
    Returns
    -------
    : int
        1 if above_pose is above the below_pose within a radial distance, 0 otherwise.
    """
    o1_x = above_pose.position.x
    o1_y = above_pose.position.y
    o1_z = above_pose.position.z

    o2_x = below_pose.position.x
    o2_y = below_pose.position.y
    o2_z = below_pose.position.z

    if axis == "x":
        if o1_x > o2_x:
            distance = np.linalg.norm(
                np.array([o1_y, o1_z]) - np.array([o2_y, o2_z]))
        else:
            return 0
    if axis == "y":
        if o1_y > o2_y:
            distance = np.linalg.norm(
                np.array([o1_x, o1_z]) - np.array([o2_x, o2_z]))
        else:
            return 0
    if axis == "z":
        if o1_z > o2_z:
            distance = np.linalg.norm(
                np.array([o1_x, o1_y]) - np.array([o2_x, o2_y]))
        else:
            return 0

    if distance < threshold_distance:
        return 1
    else:
        return 0


def proximity(object1_pose, object2_pose, threshold_distance):
    """
    Determines whether or not are in proximity with each other.
    Parameters
    ----------
    object1_pose : geometry_msgs/Pose
        The upright pose.
    object2_pose : geometry_msgs/Pose
        Current pose.
    threshold_distance : float/int
        Threshold distance within two objects are in proximity.
    Returns
    -------
    : int
        1 if within distance (in proximity), 0 otherwise.
    """
    o1_x = object1_pose.position.x
    o1_y = object1_pose.position.y
    o1_z = object1_pose.position.z

    o2_x = object2_pose.position.x
    o2_y = object2_pose.position.y
    o2_z = object2_pose.position.z

    distance = np.linalg.norm(
        np.array([o1_x, o1_y, o1_z]) - np.array([o2_x, o2_y, o2_z]))

    if distance < threshold_distance:
        return 1
    else:
        return 0


def planar(object_pose, reference_position, threshold_distance, direction="positive", axis="z"):
    """
    Determines whether or an object's pose positively or negatively distance away from a reference_position and threshold distance along the given value.
    Parameters
    ----------
    object_pose : geometry_msgs/Pose
        The object's pose.
    reference_position : float/int
        The reference or starting position to compare an objects height distance against the threshold_distance along the given axis.
     threshold_distance : float/int
        Threshold distance away from to evaluate. 
    axis : str
        Axis against which to measure deviation.
    Returns
    -------
    : int
        1 if above/below threshold distance, 0 otherwise.
    """
    if axis == "x":
        object_height = object_pose.position.x
    if axis == "y":
        object_height = object_pose.position.y
    if axis == "z":
        object_height = object_pose.position.z

    difference = fabs(object_height - reference_position)
    if direction == "positive":
        if object_height >= reference_position and difference >= threshold_distance:
            return 1
        else:
            return 0
    elif direction == "negative":
        if object_height <= reference_position and difference >= threshold_distance:
            return 1
        else:
            return 0
    else:
        raise ValueError(
            "'direction' parameter must either be 'positive' or 'negative'")