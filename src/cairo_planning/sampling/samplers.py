"""
Class for different sampling strategies.
"""
import random

import numpy as np

__all__ = ['UniformSampler']


class UniformSampler():

    """
    Uniformly samples at random each dimension given the provided limits.

    """

    def sample(self, dimension_limits):
        """
        Samples a random sample.

        Returns:
            list: Random sample.
        """
        return [random.uniform(limit[0], limit[1]) for limit in dimension_limits]


class HyperballSampler():

    def __init__(self, fraction_uniform=.05):
        self.fraction_uniform = fraction_uniform

    def _random_q(self, dimension_limits):
        return np.array([random.uniform(limit[0], limit[1]) for limit in dimension_limits])

    def _random_point_within_ball(self, centroid, radius, dimension_limits):
        random_q = self._random_q(dimension_limits)
        vector_from_centroid = random.random() * radius * (random_q -
                                                           np.array(centroid)) / np.linalg.norm(random_q - np.array(centroid))
        vector_in_ball = np.array(centroid) + vector_from_centroid
        return vector_in_ball

    def _within_limits(self, sample, limits):
        for idx, limit in enumerate(limits):
            if sample[idx] < limit[0] or sample[idx] > limit[1]:
                return False
        return True

    def sample(self, dimension_limits, centroid, radius):
        count = 1
        within_limits = False
        while not within_limits:
            count += 1
            if random.random() > self.fraction_uniform:
                sample = self._random_point_within_ball(
                    centroid, radius, dimension_limits)
            else:
                sample = self._random_q(dimension_limits)
            within_limits = self._within_limits(sample, dimension_limits)
            if within_limits:
                return sample


class DistributionSampler():

    def __init__(self, distribution_model, fraction_uniform=.1):
        """
        Samples from a fitted model that represents the distribution of discrete points. This could be a keyframe distribution, trajectory distribution, or any arbitrary distrubtion. Sampling checks if the values are within limits (usually the joint limits of the robot) passed in as an argument ot the sample function.

        Args:
            distribution_model (object): The distribution model. Expects a sample() member function.
        """
        self.model = distribution_model
        print(self.model)
        self.fraction_uniform = fraction_uniform
    
    def sample(self, dimension_limits):
        count = 1
        within_limits = False
        while not within_limits:
            count += 1
            if random.random() > self.fraction_uniform:
                sample = self.model.sample()
            else:
                sample = self._uniform_random_q(dimension_limits)
            within_limits = self._within_limits(sample, dimension_limits)
            if within_limits:
                return sample
            if count >= 10000:
                raise RuntimeError(
                    "Could not effectively sample a single point within the joint limits after 10000 attempts.")
 
    def _within_limits(self, sample, limits):
        for idx, limit in enumerate(limits):
            if sample[idx] < limit[0] or sample[idx] > limit[1]:
                return False
        return True
    
    def _uniform_random_q(self, dimension_limits):
        return np.array([random.uniform(limit[0], limit[1]) for limit in dimension_limits])


   


class GaussianSampler():

    """
    TODO: Still a work in progress. Might require a distance function and awareness of objects.

    Attributes:
        covariance (TYPE): Description
        mean (TYPE): Description
    """

    def __init__(self, mean, covariance):
        """Summary

        Args:
            mean (TYPE): Description
            covariance (TYPE): Description
        """
        self.mean = mean
        self.covariance = covariance

    def sample():
        """Summary
        """
        pass


if __name__ == "__main__":
    joint_names = ['right_j0', 'right_j1', 'right_j2',
                   'right_j3', 'right_j4', 'right_j5', 'right_j6']
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

    sampler = HyperballSampler()
    sample = sampler.sample([0, 0], .5, [(-100, 100), (-100, 100)])
    print(sample)
    print(np.linalg.norm(np.array(sample) - np.array([0, 0])) < .5)
