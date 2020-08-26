__all__ = ['STOMP']
import numpy as np
import pybullet as p

from cairo_simulator.core.link import get_link_pairs, get_joint_info_by_name, get_movable_links

from cairo_planning.collisions import self_collision_test, DisabledCollisionsContext
from cairo_planning.local.interpolation import parametric_lerp

class STOMP():

    def __init__(self, sim, robot, link_pairs, obstacles,
                 start_state_config, goal_state_config, N, max_iterations = 500, K = 5, h = 10):
        # Sim and robot specific initializations
        self.sim = sim
        self.robot = robot
        self.robot_id = self.robot.get_simulator_id()
        self.link_pairs = link_pairs
        self.obstacle_ids = [obstacle.get_simulator_id() for obstacle in obstacles]

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
        self.check_distance = 3
        self.convergence_diff = 0.1
        self.closeness_penalty = 0.05

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
            with DisabledCollisionsContext(self.sim):

                pass

    def state_cost(self, joint_configuration):
        return self._collision_cost(joint_configuration)

    def _collision_cost(self, joint_configuration):
        with DisabledCollisionsContext(self.sim):
            collision_cost = self._self_collision_cost(list(joint_configuration), self.robot,
                                                      self.link_pairs, self.closeness_penalty)
            collision_cost += self._obstacle_collision_cost(list(joint_configuration), self.robot,
                                                            get_movable_links(self.robot_id), self.obstacle_ids,
                                                            self.closeness_penalty)
        return collision_cost

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

    def _obstacle_collision_cost(self, joint_configuration, robot, links, obstacle_ids, closeness_penalty):
        robot_id = robot.get_simulator_id()

        # Set new configuration and get link states
        for i, idx in enumerate(robot._arm_dof_indices):
            p.resetJointState(robot._simulator_id, idx, targetValue=joint_configuration[i], targetVelocity=0,
                              physicsClientId=0)
        max_cost = -np.inf
        for obstacle in obstacle_ids:
            for link in links:
                closest_point_object = p.getClosestPoints(bodyA=robot_id, bodyB=obstacle, distance=self.check_distance,
                                              linkIndexA=link, physicsClientId=0)
                if closest_point_object:
                    distance = closest_point_object[0][8]
                    print("Obstacle distance = ", distance)
                    cost = closeness_penalty - distance
                    if max_cost < cost:
                        max_cost = cost

        return max_cost

    def _self_collision_cost(self, joint_configuration, robot, link_pairs, closeness_penalty):
        """
        Given a joint configuration, finds a cost which is inversely proportional to the minimum distance between all the link pairs

        It sets the robot state to the test configuration. The closest distance is calculated for every link pair.
        A cost which is inversely proportional to the minimum distance found amongst all the link pairs is returned.

        Args:
            joint_configuration (list): The joint configuration to test for self-collision
            robot (Manipulator Class instance): Manipulator Class instance
            TODO complete args
        Returns:
            float: The cost of collision
        """
        robot_id = robot.get_simulator_id()

        # Set new configuration and get link states
        for i, idx in enumerate(robot._arm_dof_indices):
            p.resetJointState(robot._simulator_id, idx, targetValue=joint_configuration[i], targetVelocity=0,
                              physicsClientId=0)
        max_cost = -np.inf
        for link1, link2 in link_pairs:
            if link1 != link2:
                closest_point_object = p.getClosestPoints(bodyA=robot_id, bodyB=robot_id, distance=self.check_distance,
                                  linkIndexA=link1, linkIndexB=link2, physicsClientId=0)
                if closest_point_object:
                    distance = closest_point_object[0][8]
                    # print("Link distance = ", distance)
                    cost = closeness_penalty - distance
                    if max_cost < cost:
                        max_cost = cost
        return max_cost



