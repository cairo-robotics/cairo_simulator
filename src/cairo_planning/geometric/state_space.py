from cairo_planning.sampling.samplers import UniformSampler


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

    def __init__(self, sampler, limits):
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
        self.limits = limits

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

    def sample(self, joint_names=None):
        if joint_names == None:
            joint_names = ['right_j0', 'right_j1', 'right_j2',
                           'right_j3', 'right_j4', 'right_j5', 'right_j6']
        selected_limits = self._get_limits(joint_names)
        return self.sampler.sample(selected_limits)
