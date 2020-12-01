from sklearn.neighbors import KernelDensity


class KernelDensityDistribution():
    """
    Wrapper around around sklearn's KernelDensity model. Will bias the process of sampling points for constrained motion planning based on demonstrated trajectories.
    
    Args:
        bandwidth (float): Parameter for KDE models that determines the spead or variance of the kernel. The smaller the bandwidth, the more closely the KDE model will fit the training data.
    """
    def __init__(self, bandwidth=.1):
        self.model = KernelDensity(kernel='gaussian', bandwidth=bandwidth)

    def fit(self, X):
        self.model.fit(X)
    
    def sample(self):
        return self.model.sample(1)[0]
