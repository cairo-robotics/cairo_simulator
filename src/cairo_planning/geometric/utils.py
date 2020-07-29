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


def quat2euler(x, y, z, w):

    roll = math.atan2(2 * y * w + 2 * x * z, 1 - 2 * y * y - 2 * z * z)
    pitch = math.atan2(2 * x * w + 2 * y * z, 1 - 2 * x * x - 2 * z * z)
    yaw = math.asin(2 * x * y + 2 * z * w)

    return np.rad2deg(roll), np.rad2deg(pitch), np.rad2deg(yaw)


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