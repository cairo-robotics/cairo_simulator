import numpy as np
from cairo_planning.geometric.tsr import TSR, TSRChain

if __name__ == "__main__":
    T0_w = TSR.xyzrpy_to_trans([0.7, 0, 0, 0, 0, 1.5708])
    Tw_e = np.array([[1., 0., 0., 0],  # no offset in x
                    [0., 1., 0., 0.],  # no offset in y
                    [0., 0., 1., 0.08],  # height above table height
                    [0., 0., 0., 1.]])
    Bw = np.zeros((6, 2))
    Bw[2, :] = [0.0, 0.02]  # Allow a little vertical movement
    Bw[5, :] = [-np.pi, np.pi]  # Allow any orientation about the z-axis of the glass
    print(Bw)
    T0_s = np.array([[-0.31974669, 0.44384044, 0.8371187, 0.11415891],
                    [0.16555036, -0.84373956,  0.5105846,  -0.14628492],
                    [0.93292825,  0.30184304,  0.19630499,  1.72863896],
                    [0.0, 0.0, 0.0, 1.0]])

    tsr = TSR(T0_w=T0_w, Tw_e=Tw_e, Bw=Bw,
                 manipindex=0, bodyandlink=16)

    print(tsr.distance(T0_s))
