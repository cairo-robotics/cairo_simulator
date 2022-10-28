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
        return self.model.predict(sample)[0]


##############################
# Helper Classes & Functions #
##############################

class FoliationTrace():
    
    def __init__(self, foliation_model):
        self.f_model = foliation_model
    
    def trace_trajectory(self, traj):
        # returns the unique model component IDs 
        return list(set([self.f_model.predict([x]) for x in traj]))
    
    def trace_trajectory_multiple(self, trajectories):
        results = []
        for traj in trajectories:
            results = results + self.trace_trajectory(traj)
        return list(set(results))
    

def winner_takes_all(X, foliation_model):
    results = [foliation_model.predict([x]) for x in X]
    return max(set(results), key = results.count)