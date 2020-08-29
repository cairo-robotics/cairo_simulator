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
        self.dof = len(start_state_config)
        self.link_pairs = link_pairs # The link pairs to check self collision for
        self.obstacle_ids = obstacles

        # Start and goal state in joint space
        self.start_state_config = np.array(start_state_config) if isinstance(start_state_config, list) else start_state_config
        self.goal_state_config = np.array(goal_state_config) if isinstance(goal_state_config, list) else goal_state_config

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
        self.convergence_diff = 0.01
        self.closeness_penalty = 0.05
        self.control_coefficient = 0.05
        self.debug = True

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
        self.trajectory_cost = self._compute_trajectory_cost()
        for iteration in range(self.max_iterations):
            K_noises, K_noisy_trajectories = self._create_K_noisy_trajectories()
            P = self._compute_probabilities(K_noisy_trajectories)
            self._update_trajectory(P, K_noises)
            new_cost = self._compute_trajectory_cost()
            if self.trajectory_cost - new_cost < self.convergence_diff and self.trajectory_cost - new_cost > 0:
                print("Feasible path found in {} iterations ...".format(iteration))
                return
            self.trajectory_cost = new_cost
        print("Reached maximum iterations without sufficient cost improvement...")

    def get_trajectory(self, time_between_each_waypoint = 5):
        return [(time_between_each_waypoint * (i+1), list(waypoint)) for i, waypoint in enumerate(self.trajectory)]

    def visualize_trajectory(self):
        # The visualization or the forward kinematics overestimates the Z axis for some unknown reason
        for i, configuration in enumerate(self.trajectory):
            end_effector_world_position = self.robot.solve_forward_kinematics(configuration)[0][0]
            p.addUserDebugText(text=str(i), textPosition=end_effector_world_position, textColorRGB=[1.0,0.0,0.0], textSize=1.0)

    def _state_cost(self, joint_configuration):
        return self._collision_cost(joint_configuration)

    def _collision_cost(self, joint_configuration):
        collision_cost = self._self_collision_cost(list(joint_configuration), self.robot,
                                                  self.link_pairs, self.closeness_penalty)
        collision_cost += self._obstacle_collision_cost(list(joint_configuration), self.robot,
                                                        get_movable_links(self.robot_id), self.obstacle_ids,
                                                        self.closeness_penalty)
        return collision_cost

    def _create_K_noisy_trajectories(self):
        # Adding trajectory + noise
        # Performing the sampling and update for one joint at a time as mentioned in the literature
        K_noisy_trajectories = []
        K_noises = []
        # TODO: Saving K best noisy trajectories so far
        for k in range(self.K):
            # single_noise is for one entire trajectory so in the end it will have a shape (waypoints x joints)
            single_noise = []
            for i in range(self.dof):
                # joint_i_noise has length equals to the N (waypoints)
                joint_i_noise = np.random.multivariate_normal(np.zeros((self.N)), self.M)
                # TODO: Clipping beyond joint limits
                single_noise.append(joint_i_noise)
            # Transposing because we want rows as waypoints and columns as joints
            single_noise = np.array(single_noise).transpose()
            single_noisy_trajectory = self.trajectory + single_noise
            K_noisy_trajectories.append(single_noisy_trajectory.copy())
            K_noises.append(single_noise.copy())
        return K_noises, K_noisy_trajectories

    def _compute_probabilities(self, K_noisy_trajectories):
        S = np.zeros((self.K, self.N))
        P = np.zeros((self.K, self.N))
        exp_values = np.zeros((self.K, self.N))

        with DisabledCollisionsContext(self.sim):
            for k in range(self.K):
                for i in range(self.N):
                    S[k][i] = self._state_cost(K_noisy_trajectories[k][i])

        S_max_for_each_i = np.max(S, axis=0)
        S_min_for_each_i = np.min(S, axis=0)

        for k in range(self.K):
            for i in range(self.N):
                exp_values[k][i] = np.exp((self.h*-1*(S[k][i] - S_min_for_each_i[i]))/
                                          (S_max_for_each_i[i] - S_min_for_each_i[i]))

        sum_exp_values_for_each_i = np.sum(exp_values, axis=0)
        for k in range(self.K):
            for i in range(self.N):
                P[k][i] = exp_values[k][i]/sum_exp_values_for_each_i[i]
        return P

    def _update_trajectory(self, P, K_noises):
        # TODO: Shouldn't update the start and the goal waypoints
        delta_trajectory = np.zeros_like(self.trajectory)
        for i in range(self.N):
            delta_trajectory_i = np.zeros((self.dof))
            for k in range(self.K):
                delta_trajectory_i += P[k][i] * K_noises[k][i]
            delta_trajectory[i] = delta_trajectory_i

        # Smoothening the delta trajectory updates by projecting onto basis vector R^-1
        smoothened_delta_trajectory = np.matmul(self.M, delta_trajectory)

        # Updating the trajectories while keeping the start and goal waypoints unchanged
        start_state_configuration = self.trajectory[0].copy()
        goal_state_configuration = self.trajectory[-1].copy()
        self.trajectory += smoothened_delta_trajectory
        self.trajectory[0] = start_state_configuration
        self.trajectory[-1] = goal_state_configuration


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
                    # print("Obstacle distance = ", distance)
                    cost = closeness_penalty - distance
                    if max_cost < cost:
                        max_cost = cost
        # TODO: velocity of the link is not multiplied as in the literature
        return max_cost

    def _self_collision_cost(self, joint_configuration, robot, link_pairs, closeness_penalty):
        """
        Given a joint configuration, finds a cost which is inversely proportional to the minimum distance between all the link pairs

        It sets the robot state to the test configuration. The closest distance is calculated for every link pair.
        A cost which is inversely proportional to the minimum distance found amongst all the link pairs is returned.

        Args:
            joint_configuration (list): The joint configuration to test for self-collision
            robot (Manipulator Class instance): Manipulator Class instance
            TODO: complete args
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

    def _compute_trajectory_cost(self):
        cost = 0
        with DisabledCollisionsContext(self.sim):
            for i in range(self.N):
                cost += self._state_cost(self.trajectory[i])
        state_cost = cost

        # Adding the control cost / acceleration cost for each DOF separately as mentioned in the literature
        for j in range(self.dof):
            control_cost_for_joint_j = self.control_coefficient * np.matmul(
                np.matmul(self.trajectory[:,j].reshape((1, self.N)), self.R),
                                    self.trajectory[:,j].reshape((self.N, 1)))
            cost += control_cost_for_joint_j[0][0]
        if self.debug:
            print("State Cost = ", state_cost)
            print("Control Cost = ", cost - state_cost)
            print("Total Cost = ", cost)
            print("")
        return cost




