import numpy as np

import math


def unit_vector(vector):
    """
    Calculates the unit vector of a vector.
    Parameters
    ----------
    vector : array-like
        Input vector.
    Returns
    -------
    : array-like
        The unit vector of the input vector.
    """
    return vector / np.linalg.norm(vector)


def angle_between(v1, v2):
    """
    Calculates the angle in radians between vectors.
    Parameters
    ----------
    v1 : array-like
        First vector.
    v2 : array-like
        Second vector.
    Returns
    -------
    : float
        Angle between v1 and v1 in radians.
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def wrap_to_interval(angles, lower=-np.pi):
    """
    Wraps an angle into a semi-closed interval of width 2*pi.
    By default, this interval is `[-pi, pi)`.  However, the lower bound of the
    interval can be specified to wrap to the interval `[lower, lower + 2*pi)`.
    If `lower` is an array the same length as angles, the bounds will be
    applied element-wise to each angle in `angles`.
    See: http://stackoverflow.com/a/32266181
    @param angles an angle or 1D array of angles to wrap
    @type  angles float or numpy.array
    @param lower optional lower bound on wrapping interval
    @type  lower float or numpy.array
    """
    return (angles - lower) % (2 * np.pi) + lower


def geodesic_error(t1, t2):
    """
    Computes the error in global coordinates between two transforms.
    @param t1 current transform
    @param t2 goal transform
    @return a 4-vector of [dx, dy, dz, solid angle]
    """ 
    t_rel = np.dot(np.linalg.inv(t1), t2)
    t2_t = np.transpose(t2)
    R = np.dot(t1, t2_t)
    angle = np.arccos((np.trace(R) - 1) / 2)
    trans = np.dot(t1[0:3, 0:3], t_rel[0:3, 3])
    return np.hstack((trans, angle))


def geodesic_distance(t1, t2, r=1.0):
    """
    Computes the geodesic distance between two transforms
    @param t1 current transform
    @param t2 goal transform
    @param r in units of meters/radians converts radians to meters
    """
    error = geodesic_error(t1, t2)
    error[3] = r * error[3]
    return np.linalg.norm(error)


def point_in_polygon(x, y, poly):
    """
    Given coordinates of closed 2D polygon, determines whether set of 2D coordinates is within polygon boundaries.

    # Source: http://www.ariel.com.au/a/python-point-int-poly.html

    Parameters
    ----------
    x : float / int
        First coordinate in the plane of the polygon
    x : float / int
        First coordinate in the plane of the polygon
    poly: list
        List of tuples (x, y) specifying boundary of polygon
    Returns
    -------
    inside : bool
        True if coordinates are inside polygon, False otherwise.
    """
    n = len(poly)
    inside = False

    p1x, p1y = poly[0]
    for i in range(n + 1):
        p2x, p2y = poly[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    return inside
