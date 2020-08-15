import numpy as np

from cairo_planning.constraints.projection import displacement, delta_x, delta_x_dist
from cairo_planning.geometric.transformation import transform_mat, bound_matrix

if __name__ == "__main__":
    To = transform_mat([.5, .5, .92], [.3, .94, 0])
    print(To)
    Tc = transform_mat([0, 0, .9], [0, 0, 0])
    print(Tc)
    C = bound_matrix([(-1000, 1000), (-1000, 1000), (0, 0)],
                     [(-.4, .4), (-.4, .4), (-1000, 1000)])
    print(C)
    Tcobj = np.matmul(np.linalg.inv(Tc), To)
    print(Tcobj)
    d = displacement(Tcobj)
    print(d)
    del_x = delta_x(d, C)
    print(del_x)
    dist = delta_x_dist(del_x)
    print(dist)
