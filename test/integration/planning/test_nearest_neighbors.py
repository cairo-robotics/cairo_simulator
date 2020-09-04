from cairo_planning.sampling.samplers import UniformSampler
from cairo_planning.geometric.state_space import SawyerConfigurationSpace
from cairo_planning.local.neighbors import NearestNeighbors

import numpy as np
from urdf_parser_py.urdf import URDF

if __name__ == "__main__":
    with open("sawyer_static.urdf", "r",  encoding="utf-8") as f:
        urdf = URDF.from_xml_string(f.read().encode())
        print(urdf.joints[6])
        joint_bounds = [[joint.name, (joint.limit.lower, joint.limit.upper)] for joint in urdf.joints if joint.limit is not None]
    print(joint_bounds)
    scs = SawyerConfigurationSpace()
    sampler = UniformSampler(scs.get_bounds())

    samples = []
    for i in range(0, 100000):
        samples.append(sampler.sample())

    nn = NearestNeighbors(X=np.array(samples), k=3, model_kwargs={"leaf_size": 40})
    for i in range(0, 5):
        dist, ind = nn.query(np.array([sampler.sample()]))
        print(dist, ind)
