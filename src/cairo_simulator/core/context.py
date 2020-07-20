from abc import ABC, abstractmethod
from functools import partial
import os
import pybullet as p
from cairo_planning.trajectory.curve import JointTrajectoryCurve
from cairo_planning.constraints.pose_contraints import orientation
from cairo_planning.local.evaluation import subdivision_evaluate
from cairo_planning.sampling import StateValidityChecker
from cairo_planning.geometric.state_space import SawyerConfigurationSpace
from cairo_planning.collisions import self_collision_test, DisabledCollisionsContext

from cairo_simulator.core.link import get_link_pairs, get_joint_info_by_name
from cairo_simulator.core.utils import ASSETS_PATH
from cairo_simulator.core.log import Logger
from cairo_simulator.devices.manipulators import Sawyer
from cairo_simulator.core.simulator import Simulator, SimObject


if os.environ.get('ROS_DISTRO'):
    import rospy


class AbstractSimContext(ABC):

    @abstractmethod
    def _setup(self, config):
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


class SawyerSimContext(AbstractSimContext):

    def __init__(self, configuration=None):
        self.configuration = configuration if configuration is not None else {}
        self._setup(self.configuration)

    def _setup(self, config):
        sim_config = config.get("sim", {
            "use_real_time": True
        })

        logger_config = config.get("logger", {
            "handlers": ['logging'],
            "level": "debug"
        })

        sawyer_config = config.get("sawyer", {
            "robot_name": "sawyer0",
            "position": [0, 0, 0.9],
            "fixed_base": True
        })

        sim_obj_configs = config.get("sim_objects", [
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
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING,0) # Disable rendering while models load
        self.sawyer_robot = Sawyer(**sawyer_config)
        self.sim_objects = [SimObject(**config)
                            for config in sim_obj_configs]
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING,1) # Turn rendering back on
        self.state_space = SawyerConfigurationSpace()
        self._setup_state_validity(self.sawyer_robot)

    def _setup_state_validity(self, sawyer_robot):
        sawyer_id = self.sawyer_robot.get_simulator_id()
        #excluded_pairs = [(get_joint_info_by_name(sawyer_id, "right_l1_2").idx, get_joint_info_by_name(sawyer_id, "right_l0").idx),
        #                  (get_joint_info_by_name(sawyer_id, "right_l1_2").idx, get_joint_info_by_name(sawyer_id, "head").idx)]
        excluded_pairs = []
        link_pairs = get_link_pairs(sawyer_id, excluded_pairs=excluded_pairs)
        self_collision_fn = partial(
            self_collision_test, robot=sawyer_robot, link_pairs=link_pairs)
        self.svc = StateValidityChecker(
            self_col_func=self_collision_fn, col_func=None, validity_funcs=None)

    def get_logger(self):
        return self.logger

    def get_sim_instance(self):
        return Simulator.get_instance()

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
        return self.state_space
