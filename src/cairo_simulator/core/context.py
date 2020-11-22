from abc import ABC, abstractmethod
from functools import partial
import os

import pybullet as p

from cairo_planning.sampling import StateValidityChecker
from cairo_planning.collisions import self_collision_test

from cairo_simulator.core.link import get_link_pairs, get_joint_info_by_name
from cairo_simulator.core.log import Logger
from cairo_simulator.devices.manipulators import Sawyer
from cairo_simulator.core.simulator import Simulator, SimObject


if os.environ.get('ROS_DISTRO'):
    import rospy


class AbstractSimContext(ABC):

    @abstractmethod
    def setup(self, config):
        pass

    @abstractmethod
    def get_logger(self):
        pass

    @abstractmethod
    def get_sim_instance(self):
        pass

    @abstractmethod
    def get_sim_objects(self):
        pass

    @abstractmethod
    def get_state_validity(self):
        pass

    @abstractmethod
    def get_state_space(self):
        pass

    @abstractmethod
    def get_collision_exclusions(self):
        pass


class SawyerSimContext(AbstractSimContext):

    def __init__(self, configuration=None, setup=True, planning_context=None):
        self.config = configuration if configuration is not None else {}
        self.planning_context = planning_context
        if setup:
            self.setup()

    def setup(self, sim_overrides=None):
        sim_config = self.config.get("sim", {
            "run_parallel": False,
            "use_real_time": True,
            "use_gui": True
        })
        if sim_overrides is not None:
            for key, value in sim_overrides.items():
                sim_config[key] = value

        logger_config = self.config.get("logger", {
            "handlers": ['logging'],
            "level": "debug"
        })

        sawyer_config = self.config.get("sawyer", {
            "robot_name": "sawyer0",
            "position": [0, 0, 0.9],
            "fixed_base": True
        })

        sim_obj_configs = self.config.get("sim_objects", [
            {"object_name": "Ground",
             "model_file_or_sim_id": "plane.urdf",
             "position": [0, 0, 0]
             }])

        if os.environ.get('ROS_DISTRO'):
            rospy.init_node("CAIRO_Sawyer_Simulator")
            use_ros = True
        else:
            use_ros = False
        self.logger = Logger(**logger_config)
        self.sim = Simulator(logger=self.logger, use_ros=use_ros, **sim_config)
        self.logger.info("Simulator {} instantiated with config {}".format(self.sim, sim_config))
        # Disable rendering while models load
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
        self.sawyer_robot = Sawyer(**sawyer_config)
        self.sim_objects = [SimObject(**config)
                            for config in sim_obj_configs]
        # Turn rendering back on
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
        self._setup_state_validity(self.sawyer_robot)
        # self._setup_collision_exclusions()

    def _setup_state_validity(self, sawyer_robot):
        sawyer_id = self.sawyer_robot.get_simulator_id()
        excluded_pairs = [(get_joint_info_by_name(sawyer_id, 'right_l6').idx,
                           get_joint_info_by_name(sawyer_id, 'right_connector_plate_base').idx)]
        link_pairs = get_link_pairs(sawyer_id, excluded_pairs=excluded_pairs)
        self_collision_fn = partial(
            self_collision_test, robot=sawyer_robot, link_pairs=link_pairs)
        self.svc = StateValidityChecker(
            self_col_func=self_collision_fn, col_func=None, validity_funcs=None)

    def _setup_collision_exclusions(self):
        ground_plane = self.get_sim_objects(['Ground'])[0]
        sawyer_id = self.sawyer_robot.get_simulator_id()
        # Exclude the ground plane and the pedestal feet from disabled collisions.
        excluded_bodies = [ground_plane.get_simulator_id()]  # the ground plane
        pedestal_feet_idx = get_joint_info_by_name(
            sawyer_id, 'pedestal_feet').idx
        # The (sawyer_idx, pedestal_feet_idx) tuple the ecluded from disabled collisions.
        excluded_body_link_pairs = [(sawyer_id, pedestal_feet_idx)]
        self.collision_exclusions = {
            "excluded_bodies": excluded_bodies,
            "excluded_body_link_pairs": excluded_body_link_pairs
        }

    def get_logger(self):
        return self.logger

    def get_sim_instance(self):
        return self.sim.get_instance()

    def get_sim_objects(self, names=None):
        if names is None:
            return self.sim_objects
        else:
            return [sim_obj for sim_obj in self.sim_objects if sim_obj._name in names]

    def get_robot(self):
        return self.sawyer_robot

    def get_state_validity(self):
        return self.svc

    def get_state_space(self):
        return self.planning_context.get_state_space()

    def get_collision_exclusions(self):
        return self.collision_exclusions
