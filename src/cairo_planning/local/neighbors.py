import numpy as np
from sklearn.neighbors import KDTree


class NearestNeighbors():
    """
    Wrapper class to interface into various Nearest Neighbor models.
    
    TODO: Currently only supports KDTree. This may be all that is needed for now.

    Args:
        X (array-like): NxD array-like data. N is number of samples, D is number of dimensions
        model_type (str, optional): [description]. Determines the choice of model. Defaults to "KDTree".
        model_args (list, optional): [description]. Args to pass to the chosen model. Defaults to None.
        model_kwargs (dict, optional): [description]. Keyword args to pass to the chosen model. Defaults to None.
    """

    def __init__(self, X, model_type="KDTree", model_args=None, model_kwargs=None):
        """
        Will fit initial model upon instantiation. 
    
        Args:
            X (array-like): NxD array-like data. N is number of samples, D is number of dimensions
            model_type (str, optional): [description]. Determines the choice of model. Defaults to "KDTree".
            model_args (list, optional): [description]. Args to pass to the chosen model. Defaults to None.
            model_kwargs (dict, optional): [description]. Keyword args to pass to the chosen model. Defaults to None.
        
        Raises:
            ValueError: Error if model type not available for use 
        """
        self.available_models = ['KDTree']
        self.X = X
        if model_type not in self.available_models:
            raise ValueError(
                "{} is not a valid value for model_type. Must be one of {}".format(model_type, self.available_models))
        else:
            self.model_type = model_type
        self.model_args = model_args if model_args is not None else []
        self.model_kwargs = model_kwargs if model_kwargs is not None else {}
        self.fit()

    def fit(self):
        """
        Fits the chosen model on the current data set X.
        """
        if self.model_type == "KDTree":
            self.model = KDTree(self.X, *self.model_args, **self.model_kwargs)


    def append(self, x):
        """
        Adds x to the dataset X. Will NOT refit the model unless explicitly asked to do so via fit().
    
        Args:
            x (array-like): 1xD vector.
        """
        self.X = np.concatenate((self.X, [x]), axis=0)

    def query(self, x_test, k=3):
        """[
        Queries the fitted nearest neighbor model for k-nearest neighbors. 

        Args:
            x_test (array-like): 1xD vector test query.
            k (int): The number of neighbors.

        Returns:
            [ndarray], [ndarray]: Returns the ndarray of distances to each neighbor and ndarray of neighbor points.
        """
        distances, indices = self.model.query([x_test], k=k)
        return  distances[0], [list(self.X[idx]) for idx in indices[0]]
