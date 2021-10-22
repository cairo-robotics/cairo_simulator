from sklearn.mixture import BayesianGaussianMixture


class VGMMFoliationClustering():

    def __init__(self, estimated_foliations=2):
        self.model = BayesianGaussianMixture(n_components=estimated_foliations)

    def fit(self, X):
        self.model.fit(X)

    def predict(self, sample):
        return self.model.predict(sample)
