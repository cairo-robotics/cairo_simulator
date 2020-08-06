import numpy as np

from cairo_planning.constraints.projection import displacement, delta_x, constraint_matrix, delta_x_dist
from cairo_planning.geometric.transformation import transform_mat

if __name__ == "__main__":
    To = transform_mat([.5, .5, .01], [0, 3.14, 0])
    print(To)
    Tc = transform_mat([0, 1, 0], [0, 0, 0])
    print(Tc)
    C = constraint_matrix([(-1000, 1000), (-1000, 1000), (0, 0)], [(-6.28, 6.28), (-6.28, 6.28), (-6.28, 6.28)])
    print(C)
    Tcobj = np.matmul(np.linalg.inv(Tc), To)
    print(Tcobj)
    d = displacement(Tcobj)
    print(d)
    del_x = delta_x(d, C)
    print(del_x)
    dist = delta_x_dist(del_x)
    print(dist)