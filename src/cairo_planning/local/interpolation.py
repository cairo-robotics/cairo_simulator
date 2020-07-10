import numpy as np
from scipy import interpolate

def cumulative_distance(local_path):
    """
    Calculates the cumulative euclidean distnace sum of a sequence of vectors.
    The distance between each consecutive point is calculated and summed. 

    Args:
        local_path (float): Numpy array of vectors representing a local path.
    
    Returns:
        int: The cumulative euclidean distance.
    """
    distance = np.sum(np.sqrt(np.sum(np.diff(local_path, axis=0)**2,1)))
    return distance

def interpolate_5poly(q0, q1, tv, qd0=None, qd1=None):
    """
    This function produces a joint space trajectory qt (MxN) where the joint
    coordinates vary from q0 (1xN) to q1 (1xN).  A quintic (5th order) polynomial is used
    with default zero boundary conditions for velocity and acceleration.
    Time is assumed to vary from 0 to 1 in M (tv) steps.  Joint velocity and
    acceleration can be optionally returned as qdt (MxN) and qddt (MxN) respectively.
    The trajectory qt, qdt and qddt are MxN matrices, with one row per time step,
    and one column per joint.

    The code in this function was adopted from the Robotics Toolbox jtraj function. Copyright (C) 1993-2017, by Peter I. Corke

    Args:
        q0 (ndarray): 1xN starting configuration vector
        q1 (ndarray): 1xN ending configuration vector
        tv (ndarray or int): Timesteps
        qd0 (ndarray, optional): Initial velocity
        qd1 (ndarray, optional): Final velocity

    Returns:
        ndarray, ndarray, ndarray: MXN matrices of positions, velocities, and acclerations at each time step.
    """
    # Normalize time steps either given the number of steps (int)
    if type(tv) is list and len(tv) > 1:
        # Get the max timescale
        timescale = max(tv)
        # divide all times by the max to normalize.
        t = tv / timescale
    else:
        timescale = 1
        # % normalized time from 0 -> 1
        t = [x / (tv - 1) for x in range(0, tv)]

    q0 = q0
    q1 = q1

    if qd0 is None and qd1 is None:
        qd0 = np.zeros(np.size(q0))
        qd1 = qd0
    else:
        qd0 = qd0
        qd1 = qd1
    # compute the polynomial coefficients
    A = 6 * (q1 - q0) - 3 * (qd1 + qd0) * timescale
    B = -15 * (q1 - q0) + (8 * qd0 + 7 * qd1) * timescale
    C = 10 * (q1 - q0) - (6 * qd0 + 4 * qd1) * timescale
    E = qd0 * timescale  # as the t vector has been normalized
    F = q0

    tt = np.array([np.power(t, 5), np.power(t, 4), np.power(
        t, 3), np.power(t, 2), t, np.ones(np.size(t))])
    c = np.array([A, B, C, np.zeros(np.size(A)), E, F])
    qt = tt.T.dot(c)

    # compute velocity
    c = np.array([np.zeros(np.size(A)), 5 * A, 4 *
                  B, 3 * C,  np.zeros(np.size(A)), E])
    qdt = tt.T.dot(c) / timescale

    # compute acceleration
    c = np.array([np.zeros(np.size(A)), np.zeros(np.size(A)),
                  20 * A, 12 * B, 6 * C,  np.zeros(np.size(A))])
    qddt = tt.T.dot(c) / np.power(timescale, 2)

    return qt, qdt, qddt


def parametric_lerp(q0, q1, steps):
    """
    This function directly interpolates between the start q0 and q1, element-wise parametrically
    via the discretized interval determined by the number of steps.

    Args:
        q0 (ndarray): Numpy vector representing the starting point.
        q1 (ndarray): Numpy vector representing the ending point.
        steps (int): Number of discrete steps to take.

    Returns:
        [ndarray]: Numpy array of the interpolation between q0 and q1.
    """
    times = [x / (steps - 1)
             for x in range(0, steps)]  # % normalized time from 0 -> 1
    return np.array([t*(q1-q0) + q0 for t in times])

if __name__ == "__main__":
    path = parametric_lerp(np.array([0,0]), np.array([10, 10]), 100)
    test = np.array([[0, 1], [1, 2], [3, 3], [6, 5]])
    print(cumulative_distance(test))
    print(cumulative_distance(path))

    path = interpolate_5poly(np.array([0,0,0,0,0,0,0]), np.array([.21, .32, .53, 1.24, -2.11, -3.1, .9]), 100)
    print(cumulative_distance(path))
