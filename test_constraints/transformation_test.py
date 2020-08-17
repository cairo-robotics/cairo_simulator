import sys

import pybullet as p

from cairo_planning.geometric.transformation import xyzrpy2trans, bounds_matrix, quat2euler, euler2quat


if __name__ == "__main__":
    print(quat2euler([-0.022076, 0.016451, 0.99929, 0.025687]))
    print(p.getEulerFromQuaternion([0.016451, 0.99929, 0.025687, -0.022076]))
    print()
    print(euler2quat(quat2euler([-0.022076, 0.016451, 0.99929, 0.025687])))
    print(p.getQuaternionFromEuler(p.getEulerFromQuaternion([0.016451, 0.99929, 0.025687, -0.022076])))
    sys.exit(0)
    