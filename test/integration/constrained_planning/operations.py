import numpy as np

from cairo_planning.constraints.projection import displacements, delta_x, delta_x_dist
from cairo_planning.geometric.transformation import xyzrpy2trans, bounds_matrix

if __name__ == "__main__":
    To = xyzrpy2trans([.5, .5, .92, .94, 0, .3])
    print(To)
    Tc = xyzrpy2trans([0, 0, .9, 0, 0, 0])
    print(Tc)
    C = bounds_matrix([(-1000, 1000), (-1000, 1000), (0, 0)],
                     [(-.4, .4), (-1000, 1000), (-.4, .4)])
    print(C)
    Tcobj = np.matmul(np.linalg.inv(Tc), To)
    print(Tcobj)
    d = displacements(Tcobj)
    print(d)
    del_x = delta_x(d, C)
    print(del_x)
    dist = delta_x_dist(del_x)
    print(dist)
