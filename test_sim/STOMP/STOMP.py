__all__ = ['STOMP']
import numpy as np
from cairo_planning.collisions import self_collision_test, DisabledCollisionsContext
from cairo_planning.local.interpolation import parametric_lerp

class STOMP():

    def __init__(self, sim, robot, obstacle_ids, excluded_bodies, excluded_body_link_pairs,
                 start_state_config, goal_state_config, N, max_iterations = 500, K = 5, h = 10):
        # Sim and robot specific initializations
        self.sim = sim
        self.robot = robot
        self.obstacle_ids = obstacle_ids
        self.excluded_bodies = excluded_bodies
        self.excluded_body_link_pairs = excluded_body_link_pairs

        # Start and goal state in joint space
        self.start_state_config = start_state_config
        self.goal_state_config = goal_state_config

        # STOMP specific initializations
        self.max_iterations = max_iterations
        self.N = N
        self.trajectory = parametric_lerp(self.start_state_config, self.goal_state_config, self.N)
        self.A = self._init_A()
        self.R = np.matmul(self.A.transpose(), self.A)
        self.R_inverse = np.linalg.inv(self.R)
        self.M = np.copy(self.R_inverse)
        self._rescale_M()
        self.K = K
        self.h = h
        self.K_best_noisy_trajectories = None
        self.trajectory_cost = np.inf
        self.convergence_diff = 0.1

    def _init_A(self):
        self.A = np.zeros((self.N, self.N))
        np.fill_diagonal(self.A, 1)
        rng = np.arange(self.N - 1)
        self.A[rng + 1, rng] = -2
        rng = np.arange(self.N - 2)
        self.A[rng + 2, rng] = 1
        return self.A

    def _rescale_M(self):
        scale_factor = np.max(self.M, axis = 0)
        self.M = self.M / (self.N * scale_factor)

    def print_trajectory(self):
        print("######### Trajectory ##########")
        print(self.trajectory)

    def plan(self):
        for _ in range(self.max_iterations):
            # Disabled collisions during planning with certain exclusions in place.
            with DisabledCollisionsContext(self.sim, self.excluded_bodies, self.excluded_body_link_pairs):

                pass

    def state_cost(self, q):
        return self.obstacle_cost(q)

    def obstacle_cost(self, q):
        return 0.1

    def create_K_noisy_trajectories(self):

        # Adding trajectory + noise
        # Transposing trajectory for adding the noise via broadcasting. The result will be transposed again.
        trajectory_transpose = self.trajectory.transpose()
        K_noisy_trajectories = []
        for _ in range(self.K):
            # noise has length equals to the N (waypoints)
            noise = np.random.multivariate_normal(np.zeros((self.N)), self.M)
            #TODO Clipping beyond joint limits
            noisy_trajectory = (trajectory_transpose + noise).transpose()
            K_noisy_trajectories.append((noisy_trajectory, noise))
        return K_noisy_trajectories

    def get_trajectory(self, time_between_each_waypoint = 5):
        return [(time_between_each_waypoint * (i+1), list(waypoint)) for i, waypoint in enumerate(self.trajectory)]



