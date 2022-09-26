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
    
    NUM_SAMPLES = 5
    fraction_uniform_increments = [0, .05, .1, .15, .2, .25, .3, .35, .4, .45, .5, .55, .6, .65, .7, .75, .8, .85, .9, .95, .99, .9999, .999999, 1]

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
        "T0_w": [0.7435968575114487, 0.01432863156468001, 0.72159506554246, -2.35393403, -0.05157824, -1.54996543],
        "Tw_e": [0, 0, 0, 0, 0, 0], 
        "Bw": [[[-.02, .02], [-100, 100], [0, .05]], 
              [[-.07, .07], [-.07, .07], [-.07, .07]]]
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
        
        for fraction in fraction_uniform_increments:
            # Create the DistributionSampler and associated SawyerTSRConstrainedSpace
            state_space = SawyerTSRConstrainedSpace(robot=sawyer_robot, TSR=tsr, svc=svc, sampler=DistributionSampler(distribution_model=model, fraction_uniform=fraction), limits=None)
            
            ptime1 = time.process_time()
            count = 0
            while count != NUM_SAMPLES:
                sample = state_space.sample()
                if sample is not None:
                    count += 1
                if time.process_time() - ptime1 >= 1000:
                    print("Only sampled {} constraint compliant points.".format(count))
                    break
            ptime2 = time.process_time()
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
    
    means = []
    stds = []
    for idx, _ in enumerate(fraction_uniform_increments):
    
        means.append(map(statistics.mean, [result[idx] for result in results.items()]))
        stds.append(map(statistics.stdev, [result[idx] for result in results.items()]))
    
    plt.errorbar(fraction_uniform_increments, means, yerr=stds, linewidth=5.0)
    plt.xlabel('Fraction Uniform Sampling', fontsize=20)
    plt.xticks(fontsize=16)
    plt.ylabel('Time (s)',  fontsize=20)
    plt.yticks(fontsize=16)
    plt.suptitle('Time to Sample {} Constrained Points vs. Fraction Uniform'.format(NUM_SAMPLES), fontsize=24)
    plt.title('Glue Application Constraint'.format(NUM_SAMPLES), fontsize=22)
    plt.legend(fontsize=20)
    plt.show()

if __name__ == "__main__":
    main()
