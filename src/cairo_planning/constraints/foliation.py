from sklearn.mixture import BayesianGaussianMixture

##########
# MODELS #
##########
class VGMMFoliationClustering():

    def __init__(self, estimated_foliations=2):
        self.model = BayesianGaussianMixture(n_components=estimated_foliations)

    def fit(self, X):
        self.model.fit(X)

    def predict(self, sample):
        return self.model.predict(sample)

####################
# Helper Functions #
####################
def winner_takes_all(X, foliation_model):
    results = []
    for x in X:
        results.append(foliation_model.predict(x))
    return max(set(results), key = results.count)