import os
import json
import time
import datetime
import statistics

import pybullet as p
if os.environ.get('ROS_DISTRO'):
    import rospy
import numpy as np

from cairo_simulator.core.sim_context import SawyerTSRSimContext
from cairo_simulator.core.utils import ASSETS_PATH

from cairo_planning.geometric.state_space import SawyerTSRConstrainedSpace
from cairo_planning.geometric.distribution import KernelDensityDistribution
from cairo_planning.sampling.samplers import DistributionSampler

import matplotlib.pyplot as plt


def main():
    
    NUM_SAMPLES = 1000
    fraction_biased_increments = [1, .95, .9, .85, .8, .75, .7, .65, .6, .55, .5, .45, .4, .35, .3, .25, .2, .15, .1, .05, .04, .03, .02, .01, 0]

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
        "T0_w": [0, 0, .9, np.pi/2, 0, np.pi/2], # level end-effector pointing away from sawyer's "front"
        "Tw_e": [0, 0, 0, 0, 0, 0], 
        "Bw": [[[-100, 100], [-100, 100], [-100, 100]], 
              [[-.07, .07], [-.07, .07], [-3.14, 3.14]]] # full yaw allowed
    }

    sim_context = SawyerTSRSimContext(config)
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
        
        for fraction in fraction_biased_increments:
            # Create the DistributionSampler and associated SawyerTSRConstrainedSpace
            state_space = SawyerTSRConstrainedSpace(robot=sawyer_robot, TSR=tsr, svc=svc, sampler=DistributionSampler(distribution_model=model, fraction_uniform=1 - fraction), limits=None)
            time1 = time.time()
            ptime1 = time.perf_counter()
            count = 0
            while count != NUM_SAMPLES:
                sample = state_space.sample()
                if sample is not None:
                    count += 1
                if time.perf_counter() - ptime1 >= 1000:
                    print("Only sampled {} constraint compliant points.".format(count))
                    break
            ptime2 = time.perf_counter()
            print(ptime2 - ptime1)
            fraction_time_tuples.append((fraction, ptime2 - ptime1))

        results[subject_dir_name] = fraction_time_tuples

    # Output results to unique filename
    now = datetime.datetime.today()
    nTime = now.strftime('%Y-%m-%dT%H-%M-%S')
    results_filename = "results_" + nTime + '.json'
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), results_filename), "w") as f:
        json.dump(results, f)

    # Create plots
    plt.figure(figsize=(12, 10))

    means = list(map(statistics.mean, [[result[1][idx][1] for result in results.items()] for idx, _ in enumerate(fraction_biased_increments)]))
    stds = list(map(statistics.stdev, [[result[1][idx][1] for result in results.items()] for idx, _ in enumerate(fraction_biased_increments)]))
    
    plt.errorbar(fraction_biased_increments, means, yerr=stds, fmt='-o', ecolor='lightgray', linewidth=3, elinewidth=1, capsize=2, capthick=2)
    plt.xlabel('Fraction of Candidate Points Drawn from Biased Distribution', fontsize=18)
    plt.xticks(fontsize=16)
    plt.ylabel('Time (s)',  fontsize=18)
    plt.yticks(fontsize=16)
    plt.title('Mean Time (w/ std) to Project {} Samples \n onto Upright Orientation Constraint Manifold'.format(NUM_SAMPLES), fontsize=22)
    # plt.legend(fontsize=20)
    plt.show()

if __name__ == "__main__":
    main()
