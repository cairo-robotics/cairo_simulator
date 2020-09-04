from cairo_planning.local.interpolation import interpolate_5poly, parametric_lerp
import numpy as np

if __name__ == "__main__":

    q0 = np.array([1, 1, 1])
    q1 = np.array([2, 2, 2])

    qt, qdt, qddt = interpolate_5poly(q0, q1, 5)

    print(qt)

    print(qdt)

    print(qddt)

    q0 = np.array([1, 1.5, 1.8])
    q1 = np.array([2, 2, 2])
    print(parametric_lerp(q0, q1, 10))

