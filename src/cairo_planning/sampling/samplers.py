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
