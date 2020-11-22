"""
Class for different sampling strategies.
"""
import random

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


class DistributionSampler():
    
    def __init__(self, distribution_model):
        """
        Samples from a fitted model that represents the distribution of discrete points. This could be a keyframe distribution, trajectory distribution, or any arbitrary distrubtion. Sampling checks if the values are within limits (usually the joint limits of the robot) passed in as an argument ot the sample function.

        Args:
            distribution_model (object): The distribution model. Expects a sample() member function.
        """
        self.model = distribution_model
        
    def _within_limits(self, sample, limits):
        for idx, limit in enumerate(limits):
            if sample[idx] < limit[0] or sample[idx] > limit[1]:
                return False
        return True
        
    def sample(self, dimension_limits):
        count = 1
        within_limits = False
        while not within_limits:
            count += 1
            sample = self.model.sample()
            within_limits = self._within_limits(sample, dimension_limits)
            if within_limits:
                return sample
            if count >= 10000:
                raise RuntimeError("Could not effectively sample a single point within the joint limits after 10000 attempts.")


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
