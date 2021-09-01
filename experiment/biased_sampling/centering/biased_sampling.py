import os
import json
import time
import datetime

import pybullet as p
if os.environ.get('ROS_DISTRO'):
    import rospy
import numpy as np

from cairo_simulator.core.sim_context import SawyerCPRMSimContext
from cairo_simulator.core.utils import ASSETS_PATH

from cairo_planning.geometric.state_space import SawyerTSRConstrainedSpace
from cairo_planning.geometric.distribution import KernelDensityDistribution
from cairo_planning.sampling.samplers import DistributionSampler, UniformSampler

import matplotlib.pyplot as plt


def main():
    
    NUM_SAMPLES = 10
    fraction_uniform_increments = [0, .05, .1, .15, .2, .25, .3, .35, .4, .45, .5, .55, .6, .65, .7, .75, .8, .85, .9, .95, 1]

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
        "Tw_e": [0.5468, -0.3946, -0.2352, -0.0515036801083644, -0.0350054025532609, 3.0945880004471826],
        "Bw": [[[-.1, .1], [-.1, .1], [0, 100]],  # Centering around the Tw_e central point that keeps end-effector over target and at or above the table
              [[-.07, .07], [-.07, .07], [-.07, .07]]] # Strict orientation constraint pointed downwards
    }

    sim_context = SawyerCPRMSimContext(config)
    sawyer_robot = sim_context.get_robot()
    svc = sim_context.get_state_validity()
    tsr = sim_context.get_tsr()

    results = {}

    # Collect all joint configurations from all demonstration .json files.
    configurations = []
    data_directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    for subject_dir_name in os.listdir(data_directory):
        print("Running biased sampling test for {}".format(subject_dir_name))
        subject_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", subject_dir_name)
        for json_file in os.listdir(subject_path):
            filename = os.path.join(subject_path, json_file)
            with open(filename, "r") as f:
                data = json.load(f)
                for entry in data:
                    configurations.append(entry['robot']['joint_angle'])


        # Create a KernelDensityDistribution with those configuration points
        model = KernelDensityDistribution(bandwidth=.1)
        model.fit(np.array(configurations))
        
        fraction_time_tuples = []
        
        for fraction in fraction_uniform_increments:
            # Create the DistributionSampler and associated SawyerTSRConstrainedSpace
            state_space = SawyerTSRConstrainedSpace(robot=sawyer_robot, TSR=tsr, svc=svc, sampler=DistributionSampler(distribution_model=model, fraction_uniform=fraction), limits=None)
            
            ptime1 = time.process_time()
            count = 0
            while count != NUM_SAMPLES:
                sample = state_space.sample()
                if sample is not None:
                    count += 1
            ptime2 = time.process_time()
            print(ptime2 - ptime1)
            fraction_time_tuples.append((fraction, ptime2 - ptime1))

        results[subject_dir_name] = fraction_time_tuples

    # Output results to unique filename
    now = datetime.datetime.today()
    nTime = now.strftime('%Y-%m-%dT%H-%M-%S')
    results_filename = "results_" + nTime
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), results_filename), "w") as f:
        json.dump(results, f)

    # Create plots
    for subject, subject_data in results.items():
    
        result_times = [result[1] for result in subject_data]
        plt.plot(fraction_uniform_increments, result_times, label=subject)
    plt.xlabel('Fraction Uniform Sampling')
    plt.ylabel('Time (s)')
    plt.suptitle('Time to Sample {} Constrained Points vs. Fraction Uniform'.format(NUM_SAMPLES), fontsize=16)
    plt.title('Upright Orientation Constraint'.format(NUM_SAMPLES), fontsize=12)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
