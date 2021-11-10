from abc import ABC, abstractmethod
from functools import partial
import os

import pybullet as p
import numpy as np

from cairo_planning.sampling import StateValidityChecker, state_validity
from cairo_planning.collisions import self_collision_test, robot_body_collision_test, multi_collision_test
from cairo_planning.geometric.state_space import SawyerConfigurationSpace, SawyerTSRConstrainedSpace, DistributionSpace
from cairo_planning.geometric.transformation import xyzrpy2trans, bounds_matrix, quat2rpy

from cairo_planning.geometric.distribution import KernelDensityDistribution
from cairo_planning.sampling.samplers import UniformSampler, DistributionSampler
from cairo_planning.geometric.tsr import TSR
from cairo_planning.geometric.utils import geodesic_distance, wrap_to_interval

from cairo_simulator.core.link import get_link_pairs, get_between_body_link_pairs, get_joint_info_by_name
from cairo_simulator.core.log import Logger
from cairo_simulator.devices.manipulators import Sawyer
from cairo_simulator.core.simulator import Simulator, SimObject
from cairo_simulator.core.primitives import PrimitiveBuilder

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

    def __init__(self, configuration=None, setup=True):
        self.config = configuration if configuration is not None else {}
        if setup:
            self.setup()

    def setup(self, sim_overrides=None):
        sim_config = self.config.get("sim", {
            "run_parallel": False,
            "use_real_time": False,
            "use_gui": True
        })
        self.config["sim"] = sim_config

        if sim_overrides is not None:
            for key, value in sim_overrides.items():
                sim_config[key] = value

        logger_config = self.config.get("logger", {
            "handlers": ['logging'],
            "level": "debug"
        })
        self.config['logger'] = logger_config

        sawyer_config = self.config.get("sawyer", {
            "robot_name": "sawyer0",
            "position": [0, 0, 0.9],
            "fixed_base": True
        })
        self.config["sawyer"] = sawyer_config

        sim_obj_configs = self.config.get("sim_objects", [
            {"object_name": "Ground",
             "model_file_or_sim_id": "plane.urdf",
             "position": [0, 0, 0]
             }])
        self.config["sim_objects"] = sim_obj_configs

        primitive_configs = self.config.get("primitives", [])

        if os.environ.get('ROS_DISTRO'):
            rospy.init_node("CAIRO_Sawyer_Simulator")
            use_ros = True
        else:
            use_ros = False
        self.logger = Logger(**logger_config)
        if not Simulator.is_instantiated():
            self.sim = Simulator(logger=self.logger,
                                 use_ros=use_ros, **sim_config)
        else:
            self.sim = Simulator.get_instance()
        self.logger.info(
            "Simulator {} instantiated with config {}".format(self.sim, sim_config))
        # Disable rendering while models load
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
        self.sawyer_robot = Sawyer(**sawyer_config)
        self.sim_objects = [SimObject(**config)
                            for config in sim_obj_configs]
        primitive_builder = PrimitiveBuilder()
        self.sim_objects = self.sim_objects + [primitive_builder.build(
            config, client=self.sim.get_client_id()) for config in primitive_configs]
        # Turn rendering back on
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
        # self._setup_state_validity(self.sawyer_robot)
        # self._setup_collision_exclusions()

    def _setup_state_validity(self, sawyer_robot, state_validity_configs):
        excluded_self_collision_pairs = self._setup_excluded_self_collision_links(
            sawyer_robot, state_validity_configs.get('self_collision_exclusions', []))
        self_collision_fn = self._setup_self_collision_fn(
            sawyer_robot, excluded_self_collision_pairs)
        collision_fn = self._setup_collision_fn(sawyer_robot)
        self.svc = StateValidityChecker(
            self_col_func=self_collision_fn, col_func=collision_fn, validity_funcs=None)

    def _setup_excluded_self_collision_links(self, sawyer_robot, excluded_link_pair_config):
        sawyer_id = sawyer_robot.get_simulator_id()
        excluded_pairs = [(get_joint_info_by_name(sawyer_id, 'right_l6').idx,
                           get_joint_info_by_name(sawyer_id, 'right_connector_plate_base').idx)]
        for pair in excluded_link_pair_config:
            excluded_pairs.append((get_joint_info_by_name(sawyer_id, pair[0]).idx,
                                   get_joint_info_by_name(sawyer_id, pair[1]).idx))
        return excluded_pairs

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

    def _setup_self_collision_fn(self, sawyer_robot):
        sawyer_id = self.sawyer_robot.get_simulator_id()
        excluded_pairs = [(get_joint_info_by_name(sawyer_id, 'right_l6').idx,
                           get_joint_info_by_name(sawyer_id, 'right_connector_plate_base').idx)]
        link_pairs = get_link_pairs(sawyer_id, excluded_pairs=excluded_pairs)
        return partial(self_collision_test, robot=sawyer_robot, link_pairs=link_pairs, client_id=self.sim.get_client_id())

    def _setup_collision_fn(self, sawyer_robot):
        collision_body_ids = self.sim.get_collision_bodies()
        if len(collision_body_ids) == 0:
            return None
        collision_fns = []
        for col_body_id in collision_body_ids:
            collision_fns.append(partial(robot_body_collision_test, robot=sawyer_robot,
                                 object_body_id=col_body_id, client_id=self.sim.get_client_id()))
        return partial(multi_collision_test, robot_object_collision_fns=collision_fns)

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
        self._setup_state_validity(self.sawyer_robot)
        return self.svc

    def get_state_space(self):
        return SawyerConfigurationSpace()

    def get_collision_exclusions(self):
        return self.collision_exclusions



class SawyerBiasedSimContext(AbstractSimContext):

    def __init__(self, configuration=None, setup=True):
        self.config = configuration if configuration is not None else {}
        if setup:
            self.setup()

    def setup(self, sim_overrides=None):
        sim_config = self.config.get("sim", {
            "run_parallel": False,
            "use_real_time": False,
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

        primitive_configs = self.config.get("primitives", [])

        state_validity_configs = self.config.get("state_validity", {})

        sampling_biasing_config = self.config.get("sampling_bias", None)

        if os.environ.get('ROS_DISTRO'):
            rospy.init_node("CAIRO_Sawyer_Simulator")
            use_ros = True
        else:
            use_ros = False
        self.logger = Logger(**logger_config)
        self.sim = Simulator(logger=self.logger, use_ros=use_ros, **sim_config)
        self.logger.info(
            "Simulator {} instantiated with config {}".format(self.sim, sim_config))
        # Disable rendering while models load
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
        self.sawyer_robot = Sawyer(**sawyer_config)
        self.sim_objects = [SimObject(**config)
                            for config in sim_obj_configs]
        primitive_builder = PrimitiveBuilder()
        self.sim_objects = self.sim_objects + \
            [primitive_builder.build(config) for config in primitive_configs]
        # Turn rendering back on
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
        # self._setup_collision_context_exclusions()
        self._setup_state_validity(self.sawyer_robot, state_validity_configs)
        self._setup_sampling_biasing(sampling_biasing_config)

    def _setup_tsr(self, tsr_config):
        T0_w = xyzrpy2trans(tsr_config['T0_w'], degrees=tsr_config['degrees'])
        Tw_e = xyzrpy2trans(tsr_config['Tw_e'], degrees=tsr_config['degrees'])
        Bw = bounds_matrix(tsr_config['Bw'][0], tsr_config['Bw'][1])

        self.tsr = TSR(T0_w=T0_w, Tw_e=Tw_e, Bw=Bw,
                       manipindex=0, bodyandlink=16)

    def _setup_collision_context_exclusions(self):
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

    def _setup_state_validity(self, sawyer_robot, state_validity_configs):
        excluded_self_collision_pairs = self._setup_excluded_self_collision_links(
            sawyer_robot, state_validity_configs.get('self_collision_exclusions', []))
        self_collision_fn = self._setup_self_collision_fn(
            sawyer_robot, excluded_self_collision_pairs)
        collision_fn = self._setup_collision_fn(sawyer_robot)
        self.svc = StateValidityChecker(
            self_col_func=self_collision_fn, col_func=collision_fn, validity_funcs=None)

    def _setup_excluded_self_collision_links(self, sawyer_robot, excluded_link_pair_config):
        sawyer_id = sawyer_robot.get_simulator_id()
        excluded_pairs = [(get_joint_info_by_name(sawyer_id, 'right_l6').idx,
                           get_joint_info_by_name(sawyer_id, 'right_connector_plate_base').idx)]
        for pair in excluded_link_pair_config:
            excluded_pairs.append((get_joint_info_by_name(sawyer_id, pair[0]).idx,
                                   get_joint_info_by_name(sawyer_id, pair[1]).idx))
        return excluded_pairs

    def _setup_self_collision_fn(self, sawyer_robot, excluded_pairs):
        sawyer_id = self.sawyer_robot.get_simulator_id()
        link_pairs = get_link_pairs(sawyer_id, excluded_pairs=excluded_pairs)
        return partial(self_collision_test, robot=sawyer_robot, link_pairs=link_pairs, client_id=self.sim.get_client_id())

    def _setup_collision_fn(self, sawyer_robot):
        collision_body_ids = self.sim.get_collision_bodies()
        if len(collision_body_ids) == 0:
            return None
        collision_fns = []
        for col_body_id in collision_body_ids:
            collision_fns.append(partial(robot_body_collision_test, robot=sawyer_robot,
                                 object_body_id=col_body_id, client_id=self.sim.get_client_id()))
        return partial(multi_collision_test, robot_object_collision_fns=collision_fns)

    def _setup_sampling_biasing(self, biasing_config):
        if biasing_config is None:
            self.biased_sampling = False
        else:
            self.biased_sampling = True
            # Create a KernelDensityDistribution with those configuration points
            self.biasing_model = KernelDensityDistribution(
                bandwidth=biasing_config['bandwidth'])
            self.biasing_model.fit(np.array(biasing_config['data']))
            self.biasing_fraction_uniform = biasing_config['fraction_uniform']

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
        limits = [['right_j0', (-3.0503, 3.0503)],
                  ['right_j1', (-3.8095, 2.2736)],
                  ['right_j2', (-3.0426, 3.0426)],
                  ['right_j3', (-3.0439, 3.0439)],
                  ['right_j4', (-2.9761, 2.9761)],
                  ['right_j5', (-2.9761, 2.9761)],
                  ['right_j6', (-4.7124, 4.7124)],
                  ['right_gripper_l_finger_joint', (0.0, 0.020833)],
                  ['right_gripper_r_finger_joint',
                   (-0.020833, 0.0)],
                  ['head_pan', (-5.0952, 0.9064)]]
        if self.biased_sampling is True:
            state_space = SawyerConfigurationSpace(sampler=DistributionSampler(
                distribution_model=self.biasing_model, fraction_uniform=self.biasing_fraction_uniform))
        else: 
            state_space = SawyerConfigurationSpace(sampler=UniformSampler())
        return state_space

    def get_tsr(self):
        return self.tsr

    def get_collision_exclusions(self):
        return self.collision_exclusions

    def delete_context(self):
        self.sim.__del__()



class SawyerTSRSimContext(AbstractSimContext):

    def __init__(self, configuration=None, setup=True):
        self.config = configuration if configuration is not None else {}
        if setup:
            self.setup()

    def setup(self, sim_overrides=None):
        sim_config = self.config.get("sim", {
            "run_parallel": False,
            "use_real_time": False,
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

        primitive_configs = self.config.get("primitives", [])
        
        state_validity_configs = self.config.get("state_validity", {})


        tsr_config = self.config.get("tsr", {
            'degrees': False,
            "T0_w": [.7, 0, 0, 0, 0, 0],
            "Tw_e": [-.2, 0, 1.0, np.pi/2, 3*np.pi/2, np.pi/2],
            "Bw": [[[0, 100], [-100, 100], [-100, .3]],  # allow some tolerance in the z and y and only positve in x
                   [[-.07, .07], [-.07, .07], [-.07, .07]]]
        })

        if os.environ.get('ROS_DISTRO'):
            rospy.init_node("CAIRO_Sawyer_Simulator")
            use_ros = True
        else:
            use_ros = False
        self.logger = Logger(**logger_config)
        self.sim = Simulator(logger=self.logger, use_ros=use_ros, **sim_config)
        self.logger.info(
            "Simulator {} instantiated with config {}".format(self.sim, sim_config))
        # Disable rendering while models load
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
        self.sawyer_robot = Sawyer(**sawyer_config)
        self.sim_objects = [SimObject(**config)
                            for config in sim_obj_configs]
        primitive_builder = PrimitiveBuilder()
        self.sim_objects = self.sim_objects + \
            [primitive_builder.build(config) for config in primitive_configs]
        # Turn rendering back on
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
        self._setup_state_validity(self.sawyer_robot)
        self._setup_tsr(tsr_config)
        # self._setup_collision_exclusions()

    def _setup_tsr(self, tsr_config):
        self.projection_epsilon = tsr_config.get('epsilon', .1)
        self.projection_e_step = tsr_config.get('e_step', .25)
        T0_w = xyzrpy2trans(tsr_config['T0_w'], degrees=tsr_config['degrees'])
        Tw_e = xyzrpy2trans(tsr_config['Tw_e'], degrees=tsr_config['degrees'])
        Bw = bounds_matrix(tsr_config['Bw'][0], tsr_config['Bw'][1])

        self.tsr = TSR(T0_w=T0_w, Tw_e=Tw_e, Bw=Bw,
                       manipindex=0, bodyandlink=16)

    def _setup_state_validity(self, sawyer_robot, state_validity_configs):
        excluded_self_collision_pairs = self._setup_excluded_self_collision_links(
            sawyer_robot, state_validity_configs.get('self_collision_exclusions', []))
        self_collision_fn = self._setup_self_collision_fn(
            sawyer_robot, excluded_self_collision_pairs)
        collision_fn = self._setup_collision_fn(sawyer_robot)
        self.svc = StateValidityChecker(
            self_col_func=self_collision_fn, col_func=collision_fn, validity_funcs=None)

    def _setup_excluded_self_collision_links(self, sawyer_robot, excluded_link_pair_config):
        sawyer_id = sawyer_robot.get_simulator_id()
        excluded_pairs = [(get_joint_info_by_name(sawyer_id, 'right_l6').idx,
                           get_joint_info_by_name(sawyer_id, 'right_connector_plate_base').idx)]
        for pair in excluded_link_pair_config:
            excluded_pairs.append((get_joint_info_by_name(sawyer_id, pair[0]).idx,
                                   get_joint_info_by_name(sawyer_id, pair[1]).idx))
        return excluded_pairs

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

    def _setup_self_collision_fn(self, sawyer_robot):
        sawyer_id = self.sawyer_robot.get_simulator_id()
        excluded_pairs = [(get_joint_info_by_name(sawyer_id, 'right_l6').idx,
                           get_joint_info_by_name(sawyer_id, 'right_connector_plate_base').idx)]
        link_pairs = get_link_pairs(sawyer_id, excluded_pairs=excluded_pairs)
        return partial(self_collision_test, robot=sawyer_robot, link_pairs=link_pairs, client_id=self.sim.get_client_id())

    def _setup_collision_fn(self, sawyer_robot):
        collision_body_ids = self.sim.get_collision_bodies()
        if len(collision_body_ids) == 0:
            return None
        collision_fns = []
        for col_body_id in collision_body_ids:
            collision_fns.append(partial(robot_body_collision_test, robot=sawyer_robot,
                                 object_body_id=col_body_id, client_id=self.sim.get_client_id()))
        return partial(multi_collision_test, robot_object_collision_fns=collision_fns)

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
        limits = [['right_j0', (-3.0503, 3.0503)],
                  ['right_j1', (-3.8095, 2.2736)],
                  ['right_j2', (-3.0426, 3.0426)],
                  ['right_j3', (-3.0439, 3.0439)],
                  ['right_j4', (-2.9761, 2.9761)],
                  ['right_j5', (-2.9761, 2.9761)],
                  ['right_j6', (-4.7124, 4.7124)],
                  ['right_gripper_l_finger_joint', (0.0, 0.020833)],
                  ['right_gripper_r_finger_joint',
                   (-0.020833, 0.0)],
                  ['head_pan', (-5.0952, 0.9064)]]
        planning_space = SawyerTSRConstrainedSpace(
            sampler=UniformSampler(), limits=limits, svc=self.svc, TSR=self.tsr, robot=self.sawyer_robot, epsilon=self.projection_epsilon, e_step=self.projection_e_step)
        return planning_space

    def get_tsr(self):
        return self.tsr

    def get_collision_exclusions(self):
        return self.collision_exclusions


class SawyerBiasedTSRSimContext(AbstractSimContext):

    def __init__(self, configuration=None, setup=True):
        self.config = configuration if configuration is not None else {}
        if setup:
            self.setup()

    def setup(self, sim_overrides=None):
        sim_config = self.config.get("sim", {
            "run_parallel": False,
            "use_real_time": False,
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

        primitive_configs = self.config.get("primitives", [])

        tsr_config = self.config.get("tsr", {})

        state_validity_configs = self.config.get("state_validity", {})

        sampling_biasing_config = self.config.get("sampling_bias", None)

        if os.environ.get('ROS_DISTRO'):
            rospy.init_node("CAIRO_Sawyer_Simulator")
            use_ros = True
        else:
            use_ros = False
        self.logger = Logger(**logger_config)
        self.sim = Simulator(logger=self.logger, use_ros=use_ros, **sim_config)
        self.logger.info(
            "Simulator {} instantiated with config {}".format(self.sim, sim_config))
        # Disable rendering while models load
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
        self.sawyer_robot = Sawyer(**sawyer_config)
        self.sim_objects = [SimObject(**config)
                            for config in sim_obj_configs]
        primitive_builder = PrimitiveBuilder()
        self.sim_objects = self.sim_objects + \
            [primitive_builder.build(config) for config in primitive_configs]
        # Turn rendering back on
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
        # self._setup_collision_context_exclusions()
        self._setup_state_validity(self.sawyer_robot, state_validity_configs)
        if tsr_config != {}:
            self._setup_tsr(tsr_config)
        else:
            self.tsr = None
        self._setup_sampling_biasing(sampling_biasing_config)

    def _setup_tsr(self, tsr_config):
        T0_w = xyzrpy2trans(tsr_config['T0_w'], degrees=tsr_config['degrees'])
        Tw_e = xyzrpy2trans(tsr_config['Tw_e'], degrees=tsr_config['degrees'])
        Bw = bounds_matrix(tsr_config['Bw'][0], tsr_config['Bw'][1])

        self.tsr = TSR(T0_w=T0_w, Tw_e=Tw_e, Bw=Bw,
                       manipindex=0, bodyandlink=16)

    def _setup_collision_context_exclusions(self):
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

    def _setup_state_validity(self, sawyer_robot, state_validity_configs):
        excluded_self_collision_pairs = self._setup_excluded_self_collision_links(
            sawyer_robot, state_validity_configs.get('self_collision_exclusions', []))
        self_collision_fn = self._setup_self_collision_fn(
            sawyer_robot, excluded_self_collision_pairs)
        collision_fn = self._setup_collision_fn(sawyer_robot)
        self.svc = StateValidityChecker(
            self_col_func=self_collision_fn, col_func=collision_fn, validity_funcs=None)

    def _setup_excluded_self_collision_links(self, sawyer_robot, excluded_link_pair_config):
        sawyer_id = sawyer_robot.get_simulator_id()
        excluded_pairs = [(get_joint_info_by_name(sawyer_id, 'right_l6').idx,
                           get_joint_info_by_name(sawyer_id, 'right_connector_plate_base').idx)]
        for pair in excluded_link_pair_config:
            excluded_pairs.append((get_joint_info_by_name(sawyer_id, pair[0]).idx,
                                   get_joint_info_by_name(sawyer_id, pair[1]).idx))
        return excluded_pairs

    def _setup_self_collision_fn(self, sawyer_robot, excluded_pairs):
        sawyer_id = self.sawyer_robot.get_simulator_id()
        link_pairs = get_link_pairs(sawyer_id, excluded_pairs=excluded_pairs)
        return partial(self_collision_test, robot=sawyer_robot, link_pairs=link_pairs, client_id=self.sim.get_client_id())

    def _setup_collision_fn(self, sawyer_robot):
        collision_body_ids = self.sim.get_collision_bodies()
        if len(collision_body_ids) == 0:
            return None
        collision_fns = []
        for col_body_id in collision_body_ids:
            collision_fns.append(partial(robot_body_collision_test, robot=sawyer_robot,
                                 object_body_id=col_body_id, client_id=self.sim.get_client_id()))
        return partial(multi_collision_test, robot_object_collision_fns=collision_fns)

    def _setup_sampling_biasing(self, biasing_config):
        if biasing_config is None:
            self.biased_sampling = False
        else:
            self.biased_sampling = True
            # Create a KernelDensityDistribution with those configuration points
            self.biasing_model = KernelDensityDistribution(
                bandwidth=biasing_config['bandwidth'])
            self.biasing_model.fit(np.array(biasing_config['data']))
            self.biasing_fraction_uniform = biasing_config['fraction_uniform']

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
        limits = [['right_j0', (-3.0503, 3.0503)],
                  ['right_j1', (-3.8095, 2.2736)],
                  ['right_j2', (-3.0426, 3.0426)],
                  ['right_j3', (-3.0439, 3.0439)],
                  ['right_j4', (-2.9761, 2.9761)],
                  ['right_j5', (-2.9761, 2.9761)],
                  ['right_j6', (-4.7124, 4.7124)],
                  ['right_gripper_l_finger_joint', (0.0, 0.020833)],
                  ['right_gripper_r_finger_joint',
                   (-0.020833, 0.0)],
                  ['head_pan', (-5.0952, 0.9064)]]
        if self.biased_sampling is True:
            if self.tsr is not None:
                # Create the DistributionSampler and associated SawyerTSRConstrainedSpace
                state_space = SawyerTSRConstrainedSpace(robot=self.sawyer_robot, TSR=self.tsr, svc=self.svc, sampler=DistributionSampler(
                    distribution_model=self.biasing_model, fraction_uniform=self.biasing_fraction_uniform), limits=None)
            else:
                state_space = SawyerConfigurationSpace(sampler=DistributionSampler(
                    distribution_model=self.biasing_model, fraction_uniform=self.biasing_fraction_uniform))
        else:
            if self.tsr is not None:
                state_space = SawyerTSRConstrainedSpace(
                    sampler=UniformSampler(), limits=limits, svc=self.svc, TSR=self.tsr, robot=self.sawyer_robot)
            else:
                state_space = SawyerConfigurationSpace(sampler=UniformSampler())
        return state_space

    def get_tsr(self):
        return self.tsr

    def get_collision_exclusions(self):
        return self.collision_exclusions

    def delete_context(self):
        self.sim.__del__()
