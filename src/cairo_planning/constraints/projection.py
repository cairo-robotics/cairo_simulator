import numpy as np

def constraint_matrix(translation_limits, rotation_limits):
    x_limits = np.array(translation_limits[0])
    y_limits = np.array(translation_limits[1])
    z_limits = np.array(translation_limits[2])
    roll_limits = np.array(rotation_limits[0])
    pitch_limits = np.array(rotation_limits[1])
    yaw_limits = np.array(rotation_limits[2])
    return np.vstack([x_limits, y_limits, z_limits, roll_limits, pitch_limits, yaw_limits])
    
def displacement(Tc):
    tc = Tc[0:3, 3]
    roll = np.arctan2(Tc[3, 2], Tc[3, 3])
    pitch = np.arcsin(Tc[3, 1])
    yaw = np.arctan2(Tc[2, 1], Tc[1, 1])
    return np.hstack([tc, roll, pitch, yaw])

def delta_x(displacement, constraint_matrix):
    delta = []
    for i in range(0, displacement.shape[0]):
        cmin = constraint_matrix[i, 0]
        cmax = constraint_matrix[i, 1]
        di = displacement[i]
        print(di, cmax, cmin)
        if di > cmax:
            delta.append(di - cmax)
        elif di < cmin:
            delta.append(di - cmin)
        else:
            delta.append(0)
    return np.array(delta)

def delta_x_dist(del_x):
    return np.linalg.norm(del_x)
    
    
if __name__ == "__main__":
    print(constraint_matrix([(0, 0), (0, 0), (0, 0)], [(-1.5, 1.5), (-1.5, 1.5), (-.05, .05)]))