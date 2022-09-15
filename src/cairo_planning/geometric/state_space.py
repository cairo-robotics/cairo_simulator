import itertools
import multiprocessing as mp
from functools import partial

import numpy as np
from cairo_planning.geometric.transformation import quat2rpy


from cairo_planning.sampling.samplers import UniformSampler
from cairo_planning.constraints.projection import project_config
from cairo_planning.geometric.utils import wrap_to_interval

class R2():

    def __init__(self, limits=None, sampler=None):
        self.limits = [['x', (0, 10)], ['y', (0, 10)]
                       ] if limits is None else limits
        self.sampler = sampler if sampler is not None else UniformSampler()

    def _get_limits(self):
        """Summary

        Returns:
            TYPE: Description
        """
        return [limits[1] for limits in self.limits if limits[0]]

    def sample(self):
        return self.sampler.sample(self._get_limits())


class DistributionSpace():

    def __init__(self, sampler, limits=None):
        """
        Represents a learned distribution space. This could be a keyframe distribution, trajectory distribution, or any arbitrary distrubtion. 
        It utilizes a sampler that contains the learned distribution model.

        The limits argument passed to the sample() method of the sampler is to ensure the sampler returns a point with dimension values 
        that lie within required limits. This could be joint limits for the dimensions representing joints, reachability limits 
        for relative distances etc,.

        Args:
            sampler (object): The sampler of learned distribution.
            limits (list): Dx2. The limits of each dimension used by the sampler to ensure the sampled dimensions are valid. 
        """
        self.sampler = sampler
        self.limits = [['right_j0', (-3.0503, 3.0503)],
                       ['right_j1', (-3.8095, 2.2736)],
                       ['right_j2', (-3.0426, 3.0426)],
                       ['right_j3', (-3.0439, 3.0439)],
                       ['right_j4', (-2.9761, 2.9761)],
                       ['right_j5', (-2.9761, 2.9761)],
                       ['right_j6', (-4.7124, 4.7124)],
                       ['right_gripper_l_finger_joint', (0.0, 0.020833)],
                       ['right_gripper_r_finger_joint',
                        (-0.020833, 0.0)],
                       ['head_pan', (-5.0952, 0.9064)]] if limits is None else limits

    def _get_limits(self, joint_names):
        """
        Given a set of joint names, it extracts the limits for each according to the provided limits list during instantiation. 
        The choice of sampler may dictate different sets of limits, either to use directly as ranges for sampling, or to check 
        if the sampler produces a valid point within these limits.

        Args:
            joint_names (list): The names of the joints to use as limits.

        Returns:
            list: Returns list of tuples, representing the limits for the joint names, in order of joint names.
        """
        return [limits[1] for limits in self.limits if limits[0] in joint_names]

    def sample(self, joint_names=None):
        if joint_names == None:
            joint_names = ['right_j0', 'right_j1', 'right_j2',
                           'right_j3', 'right_j4', 'right_j5', 'right_j6']
        selected_limits = self._get_limits(joint_names)
        return self.sampler.sample(selected_limits)
    


class SawyerTSRConstrainedSpace():

    def __init__(self, sampler, svc, TSR, robot, limits=None, epsilon=.1, q_step=3, e_step=.25, iter_count=500):
        self.limits = [['right_j0', (-3.0503, 3.0503)],
                       ['right_j1', (-3.8095, 2.2736)],
                       ['right_j2', (-3.0426, 3.0426)],
                       ['right_j3', (-3.0439, 3.0439)],
                       ['right_j4', (-2.9761, 2.9761)],
                       ['right_j5', (-2.9761, 2.9761)],
                       ['right_j6', (-4.7124, 4.7124)],
                       ['right_gripper_l_finger_joint', (0.0, 0.020833)],
                       ['right_gripper_r_finger_joint',
                        (-0.020833, 0.0)],
                       ['head_pan', (-5.0952, 0.9064)]] if limits is None else limits
        self.svc = svc
        self.TSR = TSR
        self.robot = robot
        self.epsilon = epsilon
        self.q_step = q_step
        self.e_step = e_step
        self.iter_count = iter_count
        self.sampler = sampler if sampler is not None else UniformSampler()
    
    def _get_limits(self, joint_names):
        """
        Given a set of joint names, it extracts the limits for each according to the provided limits list during instantiation. The choice of sampler may dictate different sets of limits, either to use directly as ranges for sampling, or to check if the sampler produces a valid point within these limits.

        Args:
            joint_names (list): The names of the joints to use as limits.

        Returns:
            list: Returns list of tuples, representing the limits for the joint names, in order of joint names.
        """
        return [limits[1] for limits in self.limits if limits[0] in joint_names]

    def sample(self, joint_names=None):
        if joint_names == None:
            joint_names = ['right_j0', 'right_j1', 'right_j2',
                           'right_j3', 'right_j4', 'right_j5', 'right_j6']
        selected_limits = self._get_limits(joint_names)
        sample = self.sampler.sample(selected_limits)
        return self._project(sample)

    def _project(self, sample):
        if self.svc.validate(sample):
            xyz, quat = self.robot.solve_forward_kinematics(sample)[0]
            pose = xyz + list(quat2rpy(quat))
            if not all(self.TSR.is_valid(pose)):
                q_constrained = project_config(self.robot, self.TSR, np.array(sample), np.array(sample), epsilon=self.epsilon, q_step=self.q_step, e_step=self.e_step, iter_count=self.iter_count, ignore_termination_condtions=True)
                normalized_q_constrained = []
                if q_constrained is not None:
                    for value in q_constrained:
                        normalized_q_constrained.append(
                            wrap_to_interval(value))
                    if self.svc.validate(normalized_q_constrained):
                        return normalized_q_constrained
                    else:
                        return None
                else:
                    return None
            else:
                normalized_sample = []
                for value in sample:
                        normalized_sample.append(
                            wrap_to_interval(value))
                return normalized_sample


class ParallelSawyerTSRConstrainedSpace():

    def __init__(self, sampling_fn):
        self.sampling_fn = sampling_fn
    
    
    def sample(self, n_samples, joint_names=None):
        with mp.get_context("spawn").Pool(mp.cpu_count()) as p:
            tasks = [int(n_samples/mp.cpu_count()) for n in range(0, mp.cpu_count())]
            results = p.map(self.sampling_fn, tasks)
            samples = list(itertools.chain.from_iterable(results))
            return samples


class SawyerConfigurationSpace():
    """
    Very specific configuration space according to Sawyer's articulated design. Difficult to apply a generic topology space to complex articulated arm with joint limts.

    Attributes:
        bounds (list): List of joint range limits. 
    """

    def __init__(self, sampler=None, limits=None):
        self.limits = [['right_j0', (-3.0503, 3.0503)],
                       ['right_j1', (-3.8095, 2.2736)],
                       ['right_j2', (-3.0426, 3.0426)],
                       ['right_j3', (-3.0439, 3.0439)],
                       ['right_j4', (-2.9761, 2.9761)],
                       ['right_j5', (-2.9761, 2.9761)],
                       ['right_j6', (-4.7124, 4.7124)],
                       ['right_gripper_l_finger_joint', (0.0, 0.020833)],
                       ['right_gripper_r_finger_joint',
                        (-0.020833, 0.0)],
                       ['head_pan', (-5.0952, 0.9064)]] if limits is None else limits
        self.sampler = sampler if sampler is not None else UniformSampler()

    def _get_limits(self, joint_names):
        """
        Given a set of joint names, it extracts the limits for each according to the provided limits list during instantiation. The choice of sampler may dictate different sets of limits, either to use directly as ranges for sampling, or to check if the sampler produces a valid point within these limits.

        Args:
            joint_names (list): The names of the joints to use as limits.

        Returns:
            list: Returns list of tuples, representing the limits for the joint names, in order of joint names.
        """
        return [limits[1] for limits in self.limits if limits[0] in joint_names]

    def sample(self, joint_names=None, **kwargs):
        if joint_names == None:
            joint_names = ['right_j0', 'right_j1', 'right_j2',
                           'right_j3', 'right_j4', 'right_j5', 'right_j6']
        selected_limits = self._get_limits(joint_names)
        return self.sampler.sample(selected_limits, **kwargs)
