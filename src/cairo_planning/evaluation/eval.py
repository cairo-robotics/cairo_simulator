from math import dist
import os
import json
import time

from fastdtw import fastdtw
from scipy.spatial.distance import euclidean


class IPDRelaxEvaluation():

    def __init__(self, output_dir, participant, biased_planning, ip_style):
        self.output_dir = output_dir
        self.particpant = participant
        self.planning_bias = str(biased_planning) # comes in as an int
        self.ip_style = ip_style
        self.trials = []

    def add_trial(self, trial):
        self.trials.append(trial)

    def export(self):
        file_name = self._snake_case_name() + ".json"
        output_path = os.path.join(self.output_dir, file_name)
        trials_data = {}
        trials_data["participant"] = self.particpant
        trials_data["planning_bias"] = self.planning_bias
        trials_data["ip_style"] = self.ip_style
        trials_data["trials"] = []
        for trial in self.trials:
            trial_data = {}
            trial_data["path_length"] = trial.path_length
            trial_data["a2s_distance"] = trial.a2s_distance
            trial_data["success"] = trial.success
            trial_data["a2f_percentage"] = trial.a2f_percentage
            trial_data["planning_time"] = trial.planning_time
            trial_data["ip_generation_times"] = trial.ip_gen_times
            trial_data["ip_generation_types"] = trial.ip_gen_types
            trial_data["ip_tsr_distances"] = trial.ip_tsr_distances
            trial_data["trajectory"] = trial.trajectory
            trial_data["notes"] = trial.notes
            trials_data["trials"].append(trial_data)

        with open(output_path, 'w') as f:
            json.dump(trials_data, f)

    def _snake_case_name(self):
        return "{}_{}_{}".format(self.particpant, self.planning_bias, self.ip_style)


class IPDRelaxEvaluationTrial():

    def __init__(self):
        self.path_length = -1
        self.a2s_distance = -1
        self.success = False
        self.a2f_percentage = -1
        self.planning_time = -1
        self.ip_gen_times = []
        self.trajectory = []
        self.ip_gen_types = []
        self.ip_tsr_distances = []
        self.notes = "None"
        self.timers = {}

    def eval_path_length(self, trajectory):
        total_path_length = 0
        for i in range(len(trajectory) - 1):
            total_path_length += euclidean(trajectory[i], trajectory[i + 1])
        return total_path_length

    def eval_a2s(self, trajectory, gold_trajectory, ignore_theta=True):
        if ignore_theta:
            t1 = [p[0:2] for p in trajectory]
            t2 = [p[0:2] for p in gold_trajectory]
        else:
            t1 = trajectory
            t2 = gold_trajectory
        dist, _ = fastdtw(t1, t2)
        return dist

    def eval_success(self, trajectory, goal_point, epsilon=25):
        dist_xy = euclidean(trajectory[-1][:2], goal_point[:2])
        delta_theta = abs(trajectory[-1][2] - goal_point[2])
        diff_theta = abs((delta_theta + 180) % 360 - 180)
        if dist_xy < epsilon and diff_theta < 10:
            return True
        else:
            return False

    def eval_a2f(self, trajectory_segments, constraint_eval_map, constraint_ordering):
        results = []
        for idx, segment in enumerate(trajectory_segments):
            if constraint_ordering[idx] is not None:
                evaluator = constraint_eval_map.get(
                    constraint_ordering[idx], None)
                if evaluator is not None:
                    for point in segment:
                        results.append(evaluator.validate(point))
            else:
                for point in segment:
                    results.append(True)
        return sum(results) / len(results)

    def eval_tsr_distance(self, steering_point, tsr):
        distance = tsr.distance(steering_point)
        return distance

    def start_timer(self, name):
        self.timers[name] = time.perf_counter()

    def end_timer(self, name):
        tic = self.timers[name]
        toc = time.perf_counter()
        return toc - tic
