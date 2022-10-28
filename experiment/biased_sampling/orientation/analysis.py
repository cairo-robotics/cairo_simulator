import os
import json
import statistics

import pybullet as p
if os.environ.get('ROS_DISTRO'):
    import rospy
import numpy as np


import matplotlib.pyplot as plt


def main():
    
    NUM_SAMPLES = 1000
    fraction_biased_increments = [1, .95, .9, .85, .8, .75, .7, .65, .6, .55, .5, .45, .4, .35, .3, .25, .2, .15, .1, .05, .04, .03, .02, .01, 0]

    filename = "results_2022-09-26T13-38-01.json"
    file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)
    print(file_path)
    with open(file_path, 'r') as f:
        results = json.load(f)
       
    # Create plots
    plt.figure(figsize=(16, 10))

    means = list(map(statistics.mean, [[result[1][idx][1] for result in results.items()] for idx, _ in enumerate(fraction_biased_increments)]))
    stds = list(map(statistics.stdev, [[result[1][idx][1] for result in results.items()] for idx, _ in enumerate(fraction_biased_increments)]))
    
    plt.errorbar(fraction_biased_increments, means, yerr=stds, fmt='-o', ecolor='gray', linewidth=3, elinewidth=1, capsize=2, capthick=2)
    plt.xlabel('Fraction of Candidate Points Drawn from Biased Distribution', fontsize=24)
    plt.xticks(fontsize=24)
    plt.ylabel('Time (s)',  fontsize=28)
    plt.yticks(fontsize=24)
    plt.title('Mean Time (w/ std) to Project {} Samples \n onto Upright Orientation Constraint Manifold'.format(NUM_SAMPLES), fontsize=28)
    # plt.legend(fontsize=20)
    plt.show()

if __name__ == "__main__":
    main()
