"""
Certain poritions of this module are derived from the PrPy repository of the
Personal Robotics laboratory at CMU: https://github.com/personalrobotics/prpy

This especially pertains to the usage of Task Space Region representative code.

------------------------------------------------------------------------------

Copyright (c) 2016, Carnegie Mellon University
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

- Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.
- Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.
- Neither the name of Carnegie Mellon University nor the names of its
  contributors may be used to endorse or promote products derived from this
  software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.

------------------------------------------------------------------------------
"""


from itertools import product

import numpy as np
from numpy import linalg

from cairo_planning.geometric.transformation import pose2trans, pseudoinverse, analytic_xyz_jacobian, quat2rpy, rot2rpy, transform_inv
from cairo_planning.geometric.utils import wrap_to_interval as w2i
from numpy.core.numeric import identity


def project_config(manipulator, tsr, q_s, q_old, epsilon, q_step=.5, e_step=.25, iter_count=10000, wrap_to_interval=False):
    """
    This function projects a sampled configuration point down to a constraint manifold defined implicitly by a 
    Task Space Region representation. http://cs.brown.edu/courses/csci2951-k/papers/berenson12.pdf

    This is also an equivalent approach to First-Order Retraction given by Mike Stillman's '07 IROS paper: https://www.ri.cmu.edu/pub_files/pub4/stilman_michael_2006_4/stilman_michael_2006_4.pdf

    Args:
        manipulator (Manipulator): Cairo Simulator Manipulator object. 
        q_old (ndarray): Old/prior/near configuration. 
        q_s (ndarray): Current sampled configuration.
        tsr (TSR): Task Space Region object that implicitly defines a constraint manifold.
        epsilon (float): Threshold distance within which the project sample is deemed close enough to manifold.
        q_step (float): Size of step size used for EXTEND of RRT planner.
        iter_count (int): Max number of projection iterations to try for a given sample before giving up.
        e_step (float): The fractional step size of the configuration error q_error to use during the update step.

    Returns:
        [ndarray, None]: Returns the projected point, or None if the number of iterations exceeds the max  
                         or if the projected sample is not within the manipulators joint limts.
    """
    count = 0
    while True:
        count += 1
        world_pose, _ = manipulator.solve_forward_kinematics(q_s)
        trans, quat = world_pose[0], world_pose[1]
        T0_s = pose2trans(np.hstack([trans + quat]))
        # generates the task space distance and error/displacement vector
        min_distance_new, x_err = distance_from_TSR(T0_s, tsr)
        # print(min_distance_new, x_err)
        if min_distance_new < epsilon:
            return q_s  # we've reached the manifold within epsilon error
        elif count > iter_count:
            print("max iters")
            return None
        J = manipulator.get_jacobian(q_s) 
        J_t = J[0:3, :]
        # obtain the analytic jacobian
        J_rpy = analytic_xyz_jacobian(J[3:6, :], quat2rpy(quat))
        Ja = np.vstack([np.array(J_t), np.array(J_rpy)])
        try:
            delta=.01
            # J_cross = np.dot(Ja.T, np.linalg.inv(np.dot(Ja, Ja.T)))
            J_cross = np.dot(Ja.T, np.linalg.inv(np.dot(Ja, Ja.T) + delta**2*np.ones(6)))
        except np.linalg.linalg.LinAlgError:
            # likely a singular matrix error...
            return None
        q_error = np.dot(J_cross, x_err)
        q_s = q_s - e_step * q_error
        # if the displacement of the current projected configuration relative to q_old (could be q_near etc)
        # is any larger than twice the step size q_step, we discard the projection. 
        # if wrap_to_interval:
        #     q_s_new = []
        #     for val in q_s:
        #         q_s_new.append(w2i(val))
        #     q_s = np.array(q_s_new)
        if np.linalg.norm(q_s - np.array(q_old)) > 4 * q_step or not within_joint_limits(manipulator, q_s):
                return None

        # if not within_joint_limits(manipulator, q_s):
        #     return None

def within_joint_limits(manipulator, q_s):
    """
    Given the manipulator object, checks if the given sample is within the joint limits.

    TODO: Fragile in that it assumes the _arm_joint_limits indices map correctly to the indices of q_s. 
    Works for the Sawyer Manipulator class for now.

    Args:
        manipulator (Manipualtor): Cairo Simulator Manipulator object.
        q_s (ndarray): Current configuration vecot of the manipulator

    Returns:
        [bool]: True if within limits, else False
    """
    for idx, limits in enumerate(manipulator._arm_joint_limits):
        if q_s[idx] < limits[0] or q_s[idx] > limits[1]:
            return False
    return True


def distance_from_TSR(T0_s, tsr):
    """
    Given the current transform matrix T0_s representing the current sample, this function calculates
    the task space error (translation, euler) from a the given Task Space Region (TSR) according
    to the TSR's bounds. It produces the minimum distance given equivalent euler angles and returns both that
    minimum distance and the corresponding task space error vector. 

    It uses generate_equivalent_displacement_angles() in order to create the equivalent angles (+/- pi) to apply
    to the YPR portions of the displacement vector and uses distance the smallest distance.

    Args:
        T0_s (ndarray): Transformation Matrix
        tsr ([type]): Task Space Region object.

    Returns:
        float, ndarray: The min distance from a TSR and the corresponding task space error vector.
    """
    # pose of the grasp location or the pose of the object held by the hand in world coordinates
    #T0_sp = np.dot(transform_inv(tsr.Tw_e), T0_s)
    T0_sp = np.dot(T0_s, np.linalg.inv(tsr.Tw_e))
    # T0_sp in terms of the coordinates of the target frame w given by the Task Space Region tsr.
    Tw_sp = np.dot(np.linalg.inv(tsr.T0_w), T0_sp)
    # Generate the displacement vector of Tw_sp. Displacement represents the error given T0_s relative to Tw_e transform.
    disp = displacement(Tw_sp)
    # disp[0] = T0_sp[0:3, 3][0] - tsr.T0_w[0:3, 3][0]
    # disp[1] = T0_sp[0:3, 3][1] - tsr.T0_w[0:3, 3][1]
    # disp[2] = T0_sp[0:3, 3][2] - tsr.T0_w[0:3, 3][2]
    # Since there are equivalent angle displacements for rpy, generate those equivalents by added +/- PI.
    # Use the smallest delta_x_dist of the equivalency set.
    rpys = generate_equivalent_euler_angles([disp[3], disp[4], disp[5]])
    deltas = []
    deltas.append(delta_x(disp, tsr.Bw))
    for rpy in rpys:
        deltas.append(
            delta_x(np.hstack([disp[0:3], rpy[0], rpy[1], rpy[2]]), tsr.Bw))
    distances = [delta_x_dist(delta) for delta in deltas]
    return min(distances), deltas[distances.index(min(distances))]


def displacement(Tm):
    """
    Constructs the displacment vector given a relative transformation. It returns the transform 
    as [tx, ty, tz, r, p, y] that represents the dispalcement from target w in translation and euler rotations.

    Args:
        Tm (ndarray): The transformation matrix.

    Returns:
        ndarray: The displacement vector.
    """
    # Tv = [-Tm[0:3, 3][2], -Tm[0:3, 3][0], -Tm[0:3, 3][1]]
    Tv = Tm[0:3, 3]
    Rt = Tm[0:3, 0:3]
    rpy = rot2rpy(Rt)
    return np.hstack([Tv, rpy[0], rpy[1], rpy[2]])


def generate_equivalent_euler_angles(rpy):
    """
    Given rpy angles, produces a set of rpy's that are equivalent +/- pi 
    but might ultimately produce smaller distances from a TSR.

    Args:
        rpy (array-like): The roll, pitch, yaw vector.

    Returns:
        list: The cartesian product of yaws, pitches, and rolls to create a list of rpy displacements.
    """
    rolls = [rpy[0] + np.pi, rpy[0] - np.pi]
    pitches = [rpy[1] + np.pi, rpy[1] - np.pi]
    yaws = [rpy[2] + np.pi, rpy[2] - np.pi]
    return list(product(rolls, pitches, yaws))


def delta_x(displacement, constraint_bounds):
    """
    Given a vector of displacements and a bounds/constraint matrix it produces a differential vector
    that represents the distance the displacement is from the bounds dictated by the constraint bounds. 

    For each displacement value, if the value is within the limits of the respective bound, it will be 0.

    Args:
        displacement (array-like): Vector representing translation and euler displacement from a TSR.
        constraint_bounds ([type]): The bounds/constraint matrix of the TSR.

    Returns:
        ndarray: The delta / differential vector.
    """
    delta = []
    for i in range(0, displacement.shape[0]):
        cmin = constraint_bounds[i, 0]
        cmax = constraint_bounds[i, 1]
        di = displacement[i]
        if di > cmax:
            delta.append(di - cmax)
        elif di < cmin:
            delta.append(di - cmin)
        else:
            delta.append(0)
    return np.array(delta)


def delta_x_dist(del_x):
    """
    Returns the 2-norm of a vector del_x representing the deltas of each respective
    diemsnions that a displacement vector is from the TSR constraint bounds.

    Args:
        del_x (ndarray): The delta vector from the constraint bounds of a TSR.

    Returns:
        float: 2-norm
    """
    return np.linalg.norm(del_x)
