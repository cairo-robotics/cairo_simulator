__all__ = ['STOMP']

import time
import numpy as np
import pybullet as p
from utils import generate_smoothing_matrix

from cairo_simulator.core.link import get_link_pairs, get_joint_info_by_name, get_movable_links

from cairo_planning.collisions import self_collision_test, DisabledCollisionsContext
from cairo_planning.local.interpolation import parametric_lerp

class STOMP():

    def __init__(self, sim, robot, link_pairs, obstacles,
                 start_state_config, goal_state_config, N, table_id, control_coefficient=0.5,
                 debug=True, play_pause=False, max_iterations=400, K=5, h=10):
        # Sim and robot specific initializations
        self.sim = sim
        self.robot = robot
        # The joint limits are [(-3.0503, 3.0503), (-3.8183, 2.2824), (-3.0514, 3.0514),
        # (-3.0514, 3.0514), (-2.9842, 2.9842), (-2.9842, 2.9842), (-4.7104, 4.7104)]
        self.robot_id = self.robot.get_simulator_id()
        self.dof = len(start_state_config)
        self.link_pairs = link_pairs # The link pairs to check self collision for
        self.obstacle_ids = obstacles # PyBullet object IDs

        # Start and goal state in joint space
        self.start_state_config = np.array(start_state_config) if isinstance(start_state_config, list) else start_state_config
        self.goal_state_config = np.array(goal_state_config) if isinstance(goal_state_config, list) else goal_state_config

        # STOMP specific initializations
        self.max_iterations = max_iterations
        self.N = N
        self.trajectory = parametric_lerp(self.start_state_config, self.goal_state_config, self.N)
        self.previous_trajectory = self.trajectory.copy()
        # self.A = self._init_A()
        self.R, self.M = generate_smoothing_matrix(N = self.N, derivative_order = 2, dt = 1)
        # Making changes to M according to
        # https://github.com/ros-industrial/stomp_ros/blob/melodic-devel/stomp_moveit/src/update_filters/control_cost_projection.cpp
        # Zeroing out first and last rows because we don't want to update start and goal
        self.M[0], self.M[-1] = np.zeros((self.N)), np.zeros((self.N))
        self.M[0][0], self.M[-1][-1] = 1.0, 1.0
        # self.R = np.matmul(self.A.transpose(), self.A)
        # self.R_inverse = np.linalg.inv(self.R)
        # self.M = np.copy(self.R_inverse)
        # self._rescale_M()
        self.K = K # Also called num_rollouts in ROS/MoveIt implementation
        self.max_rollouts = K + K # K + previous best trajectories
        assert self.max_rollouts >= self.K
        self.previous_best_noisy_trajectories = [] # List of tuple (trajectory cost, trajectory, noise)
        self.h = h
        self.bias_threshold = [0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15] # The acceptable change per update per joint
        self.trajectory_cost = np.inf
        self.check_distance = 3 # Distance (in meters) beyond which PyBullet will not check for collision
        self.convergence_diff = 0.01
        self.closeness_penalty = 0.05
        self.control_coefficient = control_coefficient
        self.obstacle_coefficient = 2.0

        # Custom cost specific variables
        self.table_id = table_id
        self.table_vicinity_coefficient = 1.0

        # Debugging and visualization specific variables
        self.debug = debug
        self.trajectory_visualization_ids = []
        self.play_pause = play_pause
        self.finish_button_value = False

    # def _init_A(self):
    #     self.A = np.zeros((self.N, self.N))
    #     np.fill_diagonal(self.A, 1.0)
    #     rng = np.arange(self.N - 1)
    #     self.A[rng + 1, rng] = -2.0
    #     rng = np.arange(self.N - 2)
    #     self.A[rng + 2, rng] = 1.0
    #     return self.A
    #
    # def _rescale_M(self):
    #     scale_factor = np.max(self.M, axis = 0)
    #     self.M = self.M / (self.N * scale_factor)

    def print_trajectory(self):
        print("######### Trajectory ##########")
        print(self.trajectory)

    def plan(self):
        self.trajectory_cost = self._compute_trajectory_cost()
        finish_button_value = False # Only used for interactive visualization
        invalid_iterations = 0
        invalid_iterations_cost = 0

        for iteration in range(self.max_iterations):
            K_noises, K_noisy_trajectories = self._create_K_noisy_trajectories()
            P = self._compute_probabilities(K_noisy_trajectories)

            # This function only helps in the visualization of per iteration evolution of the trajectory
            # when the self.play_pause flag is set and has nothing to do with the actual planning
            self._interactive_planning_visualization()

            validity_joint_change = self._update_trajectory(P, K_noises)
            new_cost = self._compute_trajectory_cost(self.trajectory, print_costs=self.debug)
            validity_cost = True

            if new_cost > self.trajectory_cost:
                invalid_iterations_cost += 1
                validity_cost = False

            # Reverting the changes to trajectory if the changes make it worse cost wise or
            # the changes are unacceptability large in joint space
            if not validity_cost or not validity_joint_change:
                invalid_iterations += 1
                self.trajectory = self.previous_trajectory.copy()
            elif self.trajectory_cost - new_cost < self.convergence_diff:
                print("Feasible path found in {} iterations and {} valid iterations...".
                      format(iteration, iteration - invalid_iterations))
                print("{} invalid iterations due to increase in cost and {} invalid iterations due "
                      "to large joint angle changes".format(invalid_iterations_cost,
                                                            invalid_iterations - invalid_iterations_cost))
                return
            else:
                self.trajectory_cost = new_cost
        print("Reached maximum iterations without sufficient cost improvement...")

    def get_trajectory(self, time_between_each_waypoint = 5):
        return [(time_between_each_waypoint * (i+1), list(waypoint)) for i, waypoint in enumerate(self.trajectory)]

    def visualize_trajectory(self, show_only_dot = False):
        # The visualization or the forward kinematics overestimates the Z axis for some unknown reason
        for i, configuration in enumerate(self.trajectory):
            end_effector_world_position = self.robot.solve_forward_kinematics(configuration)[0][0]
            if i == 0:
                text_string, color = "Start", [0.0,1.0,0.0]
            elif i == self.N - 1:
                text_string, color = "Goal", [0.0,1.0,0.0]
            else:
                text_string, color = str(i), [1.0,0.0,0.0]
            if show_only_dot:
                text_string = "*"
            self.trajectory_visualization_ids.append(p.addUserDebugText(text=text_string,
                                                                        textPosition=end_effector_world_position,
                                                                        textColorRGB=color, textSize=0.75))

    def clear_trajectory_visualization(self):
        for i in self.trajectory_visualization_ids:
            p.removeUserDebugItem(i)

    def _interactive_planning_visualization(self):
        if self.play_pause and not self.finish_button_value:
            resume_button_id = p.addUserDebugParameter("Resume", 1, 0, 0)
            finish_button_id = p.addUserDebugParameter("Finish", 1, 0, 0)
            self.visualize_trajectory()
            while True:
                resume_button_value = bool(p.readUserDebugParameter(resume_button_id) % 2)
                self.finish_button_value = bool(p.readUserDebugParameter(finish_button_id))
                if resume_button_value or self.finish_button_value:
                    self.clear_trajectory_visualization()
                    p.removeAllUserParameters()
                    break
                time.sleep(0.01)

    def _state_cost(self, joint_configuration):
        return self._collision_cost(joint_configuration) + self._table_vicinity_cost(joint_configuration)

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
                single_noise.append(joint_i_noise)
            # Transposing because we want rows as waypoints and columns as joints
            single_noise = np.array(single_noise).transpose()
            # single_noise = np.matmul(self.M, single_noise)

            # Adding the noise while keeping the start and goal waypoints unchanged
            start_state_configuration = self.trajectory[0].copy()
            goal_state_configuration = self.trajectory[-1].copy()

            # Adding a trajectory shaped noise to the trajectory
            single_noisy_trajectory, single_clipped_noise = self._add_and_clip_noise(self.trajectory, single_noise)

            # Restoring the start and the goal waypoints because they shouldn't be changed anyways
            single_noisy_trajectory[0] = start_state_configuration
            single_noisy_trajectory[-1] = goal_state_configuration

            # Also making the noise for the start and goal 0 as they are not going to change
            single_clipped_noise[0] = np.zeros((self.dof))
            single_clipped_noise[-1] = np.zeros((self.dof))

            K_noisy_trajectories.append(single_noisy_trajectory.copy())
            K_noises.append(single_clipped_noise.copy())
        return K_noises, K_noisy_trajectories

    def _return_K_plus_previous_best_noisy_trajectories(self):
        K_noises, K_noisy_trajectories = self._create_K_noisy_trajectories()
        sorted_K_noisy_trajectories_with_noises = sorted([(self._compute_trajectory_cost(K_noisy_trajectories[k]),
                                                    K_noisy_trajectories[k], K_noises[k]) for k in range(self.K)],
                                                         key=lambda x:x[0])


    def _add_and_clip_noise(self, trajectory, single_noise):
        single_noisy_trajectory = np.zeros_like(trajectory)
        single_clipped_noise = np.zeros_like(trajectory)
        for j in range(self.dof):
            single_noisy_trajectory[:, j] = np.clip(trajectory[:, j] + single_noise[:, j],
                                                    a_min=self.robot._arm_joint_limits[j][0],
                                                    a_max=self.robot._arm_joint_limits[j][1])
            single_clipped_noise[:, j] = single_noisy_trajectory[:, j] - trajectory[:, j]
        return single_noisy_trajectory, single_clipped_noise


    def _compute_probabilities(self, K_noisy_trajectories):
        S = np.zeros((self.K, self.N))
        P = np.zeros((self.K, self.N))
        exp_values = np.zeros((self.K, self.N))

        with DisabledCollisionsContext(self.sim):
            for k in range(self.K):
                for i in range(self.N):
                    S[k][i] =  self._state_cost(K_noisy_trajectories[k][i]) + \
                               self._control_cost(K_noisy_trajectories[k])
                    # S[k][i] = self._control_cost(K_noisy_trajectories[k]) / self.N

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
        delta_trajectory = np.zeros_like(self.trajectory)
        for i in range(self.N):
            delta_trajectory_i = np.zeros((self.dof))
            for k in range(self.K):
                delta_trajectory_i += P[k][i] * K_noises[k][i]
            delta_trajectory[i] = delta_trajectory_i

        # Smoothening the delta trajectory updates by projecting onto basis vector R^-1
        smoothened_delta_trajectory = np.matmul(self.M, delta_trajectory)

        # Updating the trajectories
        # Saving the previous trajectory in case the changes make the cost worse or
        # the configuration change is more than the self.bias_threshold
        self.previous_trajectory = self.trajectory.copy()
        self.trajectory += smoothened_delta_trajectory

        # Finding if the joint angle changes are within acceptable thresholds (using self.bias_threshold)
        proposed_change = self.trajectory - self.previous_trajectory
        proposed_change_max_each_joint = np.max(proposed_change, axis=0)
        for d in range(self.dof):
            if proposed_change_max_each_joint[d] - self.bias_threshold[d] > 0:
                return False
        return True


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
        return max_cost * self.obstacle_coefficient

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

    def _compute_trajectory_cost(self, trajectory, print_costs = False):
        cost = 0
        with DisabledCollisionsContext(self.sim):
            for i in range(self.N):
                cost += self._state_cost(trajectory[i])
        state_cost = cost
        cost += self._control_cost(trajectory)

        if print_costs:
            print("State Cost = ", state_cost)
            print("Control Cost = ", cost - state_cost)
            print("Total Cost = ", cost)
            print("")
        return cost

    # Inspired from
    #  https://github.com/ros-industrial/stomp_ros/blob/melodic-devel/stomp_core/src/stomp.cpp
    def _control_cost(self, trajectory):
        cost = 0.0
        max_cost = 0.0
        # Adding the control cost / acceleration cost for each DOF separately as mentioned in the literature
        for j in range(self.dof):
            control_cost_for_joint_j = self.control_coefficient * np.matmul(
                np.matmul(trajectory[:, j].reshape((1, self.N)), self.R),
                trajectory[:, j].reshape((self.N, 1)))
            cost += control_cost_for_joint_j[0][0]
            if control_cost_for_joint_j[0][0] > max_cost:
                max_cost = control_cost_for_joint_j[0][0]
        return cost/max_cost

    def _table_vicinity_cost(self, joint_configuration):
        end_effector_world_position = self.robot.solve_forward_kinematics(joint_configuration)[0][0]
        z_coordinate = end_effector_world_position[2]
        cost = np.linalg.norm(np.array([0.74, 0.05, .55]) -
                              np.array(end_effector_world_position)) # TODO: find tabletop's center's pose programmatically
        cost *= self.table_vicinity_coefficient
        return cost/self.N