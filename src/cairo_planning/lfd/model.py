import numpy as np

class KeyframeLFDModel():
    """
    Used to represent a learned keyframe LfD model.  
    """
    
    def __init__(self):
        pass

    def import_model(self, config_file, demonstration_directory):
        pass

def sample_rank(model, samples):
    array = []
    for sample in samples:
        array.append(sample)
    np_array = np.array(array)

    scores = model.score_samples(np_array)
    order = np.argsort(-scores)
    samples = np_array[order]
    rank_sorted_sampled = np.asarray(samples)
    return rank_sorted_sampled
