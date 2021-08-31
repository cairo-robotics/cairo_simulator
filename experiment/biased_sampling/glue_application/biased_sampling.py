import os
import json
import time

import pybullet as p
if os.environ.get('ROS_DISTRO'):
    import rospy
import numpy as np

from cairo_simulator.core.sim_context import SawyerCPRMSimContext
from cairo_simulator.core.utils import ASSETS_PATH

from cairo_planning.geometric.state_space import SawyerTSRConstrainedSpace
from cairo_planning.geometric.distribution import KernelDensityDistribution
from cairo_planning.sampling.samplers import DistributionSampler, UniformSampler
from cairo_planning.geometric.transformation import xyzrpy2trans, bounds_matrix, quat2rpy
from cairo_planning.geometric.tsr import TSR

def main():
    
    config = {}
    config["sim"] = {
            "run_parallel": False,
            "use_real_time": False,
            "use_gui": False
        }
    config["sim_objects"] = [
        {
            "object_name": "Ground",
            "model_file_or_sim_id": "plane.urdf",
            "position": [0, 0, 0]
        }
    ]

    config["tsr"] = {
        'degrees': False,
        "T0_w": [0, 0, .9, 0, 0, 0], # the position should often be that of sawyer. usually 0, 0, .9 according to default sim_context configuration
        "Tw_e": [0.6687, 0.1269, -0.2352, -3.1335505133609978, -0.3884907841856089, 0.0011134329847108074],
        "Bw": [[[-100, 100], [-100, 100], [-.05, .05]],  # Tight height constraint
              [[-.07, .07], [-.07, .07], [-.07, .07]]] # Strict orientation constraint
    }

    sim_context = SawyerCPRMSimContext(config)
    sawyer_robot = sim_context.get_robot()
    svc = sim_context.get_state_validity()
    tsr = sim_context.get_tsr()
  
    # Collect all joint configurations from all demonstration .json files.
    configurations = []
    directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    for name in os.listdir(directory):
        filename = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", name)
        with open(filename, "r") as f:
            data = json.load(f)
            for entry in data:
                configurations.append(entry['robot']['joint_angle'])


    # Create a KernelDensityDistribution with those configuration points
    model = KernelDensityDistribution(bandwidth=.1)
    model.fit(np.array(configurations))
    
    fraction_time_tuples = []
    fractions = [0, .05, .1, .15, .2, .25, .3, .35, .4, .45, .5, .55, .6, .65, .7, .75, .8, .85, .9, .95, 1]
    for fraction in fractions:
        # Create the DistributionSampler and associated SawyerTSRConstrainedSpace
        state_space = SawyerTSRConstrainedSpace(robot=sawyer_robot, TSR=tsr, svc=svc, sampler=DistributionSampler(distribution_model=model, fraction_uniform=fraction), limits=None)
        
        ptime1 = time.process_time()
        count = 0
        while count != 1000:
            sample = state_space.sample()
            if sample is not None:
                count += 1
                print(sample)
        ptime2 = time.process_time()
        print(ptime2 - ptime1)
        fraction_time_tuples.append((fraction, ptime2 - ptime1))

    print(fraction_time_tuples)

if __name__ == "__main__":
    main()
