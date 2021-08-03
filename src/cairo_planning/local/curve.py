from cairo_planning.local.bezier import bezier_curve, bezier_coefficients
from cairo_planning.local.minjerk import minjerk_coefficients, minjerk_trajectory


class JointTrajectoryCurve():
    
    def __init__(self, interpolation='minjerk'):
        self.interpolation = interpolation
        
    def generate_trajectory(self, points, move_time=1, num_intervals=3):
        if self.interpolation == 'minjerk':
            m_coeff = minjerk_coefficients(points)
            minjerk_traj = minjerk_trajectory(m_coeff, num_intervals=num_intervals)
            traj_curve = list(zip([move_time * n/len(minjerk_traj) for n in range(0, len(minjerk_traj))], [list(q[0]) for q in minjerk_traj], [list(q[1]) for q in minjerk_traj], [list(q[2]) for q in minjerk_traj]))
            return traj_curve