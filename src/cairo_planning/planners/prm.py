from math import inf
import itertools
from functools import partial
import multiprocessing as mp
import random
import types
from decimal import Decimal, localcontext, ROUND_DOWN
from timeit import default_timer as timer
import time

import numpy as np
import igraph as ig

from cairo_planning.local.evaluation import subdivision_evaluate
from cairo_planning.local.interpolation import cumulative_distance
from cairo_planning.local.neighbors import NearestNeighbors
from cairo_planning.planners.parallel_workers import parallel_connect_worker, parallel_sample_worker, parallel_projection_worker, parallel_cbirrt_worker
from cairo_planning.planners.tree import CBiRRT2
from cairo_planning.planners import utils
from cairo_simulator.core.log import Logger

__all__ = ['PRM', 'PRMParallel', 'CPRM']


class PRM():
    def __init__(self, state_space, state_validity_checker, interpolation_fn, params):
        self.graph = ig.Graph()
        self.state_space = state_space
        self.svc = state_validity_checker
        self.interp_fn = interpolation_fn
        self.n_samples = params.get('n_samples', 4000)
        self.k = params.get('k', 5)
        self.ball_radius = params.get('ball_radius', .55)
        print("N: {}, k: {}, r: {}".format(
            self.n_samples, self.k, self.ball_radius))

    def plan(self, q_start, q_goal):
        # Initial sampling of roadmap and NN data structure.
        print("Initializing roadmap...")
        self._init_roadmap(q_start, q_goal)
        print("Generating valid random samples...")
        samples = self._generate_samples()
        # Create NN datastructure
        print("Creating NN datastructure...")
        self.nn = NearestNeighbors(X=np.array(
            samples), model_kwargs={"leaf_size": 100})
        # Generate NN connectivity.
        print("Generating nearest neighbor connectivity...")
        connections = self._generate_connections(samples=samples)
        print("Generating graph from samples and connections...")
        self._build_graph(samples, connections)
        print("Attaching start and end to graph...")
        self._attach_start_and_end()
        print("Finding feasible best path in graph if available...")
        if self._success():
            plan = self._smooth(self.best_sequence())
            return plan
        else:
            return []

    def best_sequence(self):
        return self.graph.get_shortest_paths('start', 'goal', weights='weight', mode='ALL')[0]

    def get_path(self, plan):
        points = [self.graph.vs[idx]['value'] for idx in plan]
        pairs = list(zip(points, points[1:]))
        segments = [self.interp_fn(np.array(p[0]), np.array(p[1]))
                    for p in pairs]
        segments = [[list(val) for val in seg] for seg in segments]
        path = []
        for seg in segments:
            path = path + seg
        return path

    def _init_roadmap(self, q_start, q_goal):
        self.graph.add_vertex("start")
        # Start is always at the 0 index.
        self.graph.vs[0]["value"] = list(q_start)
        self.graph.add_vertex("goal")
        # Goal is always at the 1 index.
        self.graph.vs[1]["value"] = list(q_goal)

    def _smooth(self, plan):
        def shortcut(plan):
            idx_range = [i for i in range(0, len(plan))]
            indexed_plan = list(zip(idx_range, plan))
            for curr_idx, vid1 in indexed_plan:
                for test_idx, vid2 in indexed_plan[::-1]:
                    p1 = self.graph.vs[vid1]["value"]
                    p2 = self.graph.vs[vid2]["value"]
                    valid, _ = self._extend(np.array(p1), np.array(p2))
                    if test_idx == curr_idx + 1:
                        break
                    if valid and curr_idx < test_idx:
                        del plan[curr_idx+1:test_idx]
                        return False, plan
            return True, plan

        finished = False
        current_plan = plan
        while not finished:
            finished, new_plan = shortcut(current_plan)
            if new_plan is None:
                finished = True
                break
            current_plan = new_plan
        return current_plan

    def _generate_samples(self):
        # sampling_times = [0]
        count = 0
        valid_samples = []
        while count <= self.n_samples:
            # start_time = timer()
            q_rand = self._sample()
            if np.any(q_rand):
                if self._validate(q_rand):
                    if count % 100 == 0:
                        print("{} valid samples...".format(count))
                    valid_samples.append(q_rand)
                    count += 1
                    # sampling_times.append(timer() - start_time)
            # print(sum(sampling_times) / len(sampling_times))
        return valid_samples

    def _generate_connections(self, samples):
        connections = []
        for q_rand in samples:
            for q_neighbor in self._neighbors(q_rand):
                valid, local_path = self._extend(
                    np.array(q_rand), np.array(q_neighbor))
                if valid:
                    connections.append(
                        [q_neighbor, q_rand, self._weight(local_path)])
        print("{} connections out of {} samples".format(
            len(connections), len(samples)))
        return connections

    def _build_graph(self, samples, connections):
        values = [self.graph.vs[0]["value"],
                  self.graph.vs[1]["value"]] + samples
        values = [list(value) for value in values]
        self.graph.add_vertices(len(values))
        self.graph.vs["value"] = values
        edges = [(self._idx_of_point(c[0]), self._idx_of_point(c[1]))
                 for c in connections]
        weights = [c[2] for c in connections]
        self.graph.add_edges(edges)
        self.graph.es['weight'] = weights

    def _attach_start_and_end(self):
        start = self.graph.vs[0]['value']
        end = self.graph.vs[1]['value']
        start_added = False
        end_added = False
        for q_near in self._neighbors(start, k_override=30, within_ball=True):
            if self._idx_of_point(q_near) != 0:
                successful, local_path = self._extend(start, q_near)
                if successful:
                    start_added = True
                    self._add_edge_to_graph(
                        start, q_near, self._weight(local_path))
        for q_near in self._neighbors(end, k_override=30, within_ball=True):
            if self._idx_of_point(q_near) != 1:
                successful, local_path = self._extend(q_near, end)
                if successful:
                    end_added = True
                    self._add_edge_to_graph(
                        q_near, end, self._weight(local_path))
        if not start_added or not end_added:
            raise Exception("Planning failure! Could not add either start {} and end {} successfully to graph.".format(
                {start_added}, {end_added}))

    def _success(self):
        paths = self.graph.shortest_paths_dijkstra(
            [0], [1], weights='weight', mode='ALL')
        if len(paths) > 0 and paths[0][0] != inf:
            return True
        return False

    def _validate(self, sample):
        return self.svc.validate(sample)

    def _extend(self, q_near, q_rand):
        local_path = self.interp_fn(np.array(q_near), np.array(q_rand))
        valid = subdivision_evaluate(self.svc.validate, local_path)
        if valid:
            return True, local_path
        else:
            return False, []

    def _neighbors(self, sample, k_override=None, within_ball=True):
        if k_override is not None:
            k = k_override
        else:
            k = self.k
        distances, neighbors = self.nn.query(sample, k=k)
        if within_ball:
            return [neighbor for distance, neighbor in zip(
                distances, neighbors) if distance <= self.ball_radius and distance > 0]
        else:
            return [neighbor for distance, neighbor in sorted(
                list(zip(distances, neighbors)), key=lambda x: x[0]) if distance > 0]

    def _sample(self):
        return np.array(self.state_space.sample())

    def _add_edge_to_graph(self, q_near, q_sample, edge_weight):
        q_near_idx = self._idx_of_point(q_near)
        q_sample_idx = self._idx_of_point(
            q_sample)
        if tuple(sorted([q_near_idx, q_sample_idx])) not in set([tuple(sorted(edge.tuple)) for edge in self.graph.es]):
            self.graph.add_edge(q_near_idx, q_sample_idx,
                                **{'weight': edge_weight})

    def _weight(self, local_path):
        return cumulative_distance(local_path)

    def _idx_of_point(self, point):
        return self.graph.vs['value'].index(list(point))


class PRMParallel():
    # TODO: Need to incorporate new name attribute as string of value for fast indexing / lookups.
    # VERY OUT OF DATE
    def __init__(self, sim_context_cls, sim_config, svc, interpolation_fn, params):
        self.graph = ig.Graph()
        self.sim_context_cls = sim_context_cls
        self.sim_config = sim_config
        self.svc = svc
        self.interp_fn = interpolation_fn
        self.n_samples = params.get('n_samples', 8000)
        self.k = params.get('k', 8)
        self.ball_radius = params.get('ball_radius', .5)
        print("N: {}, k: {}, r: {}".format(
            self.n_samples, self.k, self.ball_radius))

    def plan(self, q_start, q_goal):
        # Initial sampling of roadmap and NN data structure.
        print("\n\nInitializing roadmap...\n\n")
        self._init_roadmap(q_start, q_goal)
        # Sample in parallel
        samples = self._parallel_sample()
        print("\n\nGenerated samples {}\n\n".format(len(samples)))
        # Create NN datastructure
        self.nn = NearestNeighbors(X=np.array(
            samples), model_kwargs={"leaf_size": 50})
        # Generate NN connectivity.
        print("\n\nGenerating nearest neighbor connectivity in parallel...\n\n")
        connections = self._parallel_connect(samples=samples)
        print("\n\nGenerate graph from samples and connections...\n\n")
        self._build_graph(samples, connections)
        print("\n\nAttaching start and end to graph...\n\n")
        self._attach_start_and_end()
        print("\n\nFinding path...\n\n")
        if self._success():
            print("Found path")
            return self.best_sequence()
        else:
            return []

    def best_sequence(self):
        return self.graph.get_shortest_paths('start', 'goal', weights='weight', mode='ALL')[0]

    def get_path(self, plan):
        points = [self.graph.vs[idx]['value'] for idx in plan]
        pairs = list(zip(points, points[1:]))
        segments = [self.interp_fn(np.array(p[0]), np.array(p[1]))
                    for p in pairs]
        segments = [[list(val) for val in seg] for seg in segments]
        path = []
        for seg in segments:
            path = path + seg
        return path

    def _init_roadmap(self, q_start, q_goal):
        self.start_name = str(list(q_start))
        self.graph.add_vertex(str(list(q_start)))
        self.graph.vs.find(name=str(list(q_start)))["value"] = list(q_start)

        self.goal_name = str(list(q_goal))
        self.graph.add_vertex("goal")
        self.graph.vs.find(name=str(list(q_start)))["value"] = list(q_goal)

    def _parallel_sample(self, n_tasks=8):
        with mp.get_context("spawn").Pool(mp.cpu_count()) as p:
            tasks = [int(self.n_samples/n_tasks) for n in range(0, n_tasks)]
            worker = partial(parallel_sample_worker,
                             sim_context_cls=self.sim_context_cls, sim_config=self.sim_config)
            results = p.map(worker, tasks)
            samples = list(itertools.chain.from_iterable(results))
            return samples

    def _parallel_connect(self, samples, n_tasks=8):
        with mp.get_context("spawn").Pool(mp.cpu_count()) as p:
            tasks = []
            for q_sample in samples:
                distances, neighbors = self.nn.query(q_sample, k=self.k)
                q_neighbors = [neighbor for distance, neighbor in zip(
                    distances, neighbors) if distance <= self.ball_radius and distance > 0]
                tasks.append((q_sample, q_neighbors))
            batches = np.array_split(tasks, n_tasks)
            worker = partial(parallel_connect_worker,
                             interp_fn=self.interp_fn, distance_fn=cumulative_distance, sim_context_cls=self.sim_context_cls, sim_config=self.sim_config)
            results = p.map(worker, batches)
            connections = list(itertools.chain.from_iterable(results))
            print("{} connections out of {} samples".format(
                len(connections), len(samples)))
            return connections

    def _build_graph(self, samples, connections):
        values = [self.graph.vs.find(self.start_name)["value"],
                  self.graph.vs[self.goal_name]["value"]] + samples
        str_values = [str(list(value)) for value in values]
        values = [list(value) for value in values]
        self.graph.add_vertices(len(values))
        self.graph.vs["name"] = str_values
        self.graph.vs["value"] = values
        edges = [(self._idx_of_point(str(c[0])), self._idx_of_point(str(c[1])))
                 for c in connections]
        weights = [c[2] for c in connections]
        self.graph.add_edges(edges)
        self.graph.es['weight'] = weights

    def _attach_start_and_end(self):
        start = self.graph.vs[self.start_name]['value']
        end = self.graph.vs[self.goal_name]['value']
        start_added = False
        end_added = False
        for q_near in self._neighbors(start, k_override=30, within_ball=True):
            if self._idx_of_point(q_near) != 0:
                successful, local_path = self._extend(start, q_near)
                if successful:
                    start_added = True
                    self._add_edge_to_graph(
                        str(start), str(q_near), self._weight(local_path))
                    break
        for q_near in self._neighbors(end, k_override=30, within_ball=True):
            if self._idx_of_point(q_near) != 1:
                successful, local_path = self._extend(q_near, end)
                if successful:
                    end_added = True
                    self._add_edge_to_graph(
                        str(q_near), str(end), self._weight(local_path))
                    break
        if not start_added or not end_added:
            raise Exception("Planning failure! Could not add either start {} and end {} successfully to graph.".format(
                {start_added}, {end_added}))

    def _success(self):
        paths = self.graph.shortest_paths_dijkstra(
            [0], [1], weights='weight', mode='ALL')
        if len(paths) > 0 and paths[0][0] != inf:
            return True
        return False

    def _extend(self, q_near, q_rand):
        local_path = self.interp_fn(np.array(q_near), np.array(q_rand))
        valid = subdivision_evaluate(self.svc.validate, local_path)
        if valid:
            return True, local_path
        else:
            return False, []

    def _add_edge_to_graph(self, q_near, q_sample, edge_weight):
        q_near_idx = self._idx_of_point(q_near)
        q_sample_idx = self._idx_of_point(
            q_sample)
        self.graph.add_edge(q_near_idx, q_sample_idx,
                            **{'weight': edge_weight})

    def _weight(self, local_path):
        return cumulative_distance(local_path)

    def _neighbors(self, sample, within_ball=True):
        distances, neighbors = self.nn.query(sample, k=self.k)
        if within_ball:
            return [neighbor for distance, neighbor in zip(
                distances, neighbors) if distance <= self.ball_radius and distance > 0]
        else:
            return [neighbor for distance, neighbor in sorted(
                list(zip(distances, neighbors)), key=lambda x: x[0]) if distance > 0]

    def _idx_of_point(self, name):
        return self.graph.vs[name].index


class CPRM():

    def __init__(self, sim_context_cls, sim_config, robot, tsr, state_space, tree_state_space, state_validity_checker, interpolation_fn, params, tree_params, logger=None):
        self.sim_context = sim_context_cls
        self.sim_config = sim_config
        self.robot = robot
        self.tsr = tsr
        self.state_space = state_space
        self.tree_state_space = tree_state_space
        self.svc = state_validity_checker
        self.interp_fn = interpolation_fn
        self.n_samples = params.get('n_samples', 600)
        self.k = params.get('k', 5)
        self.ball_radius = params.get('ball_radius', .55)
        self.tree_params = tree_params
        self.smooth_path = params.get('smooth_path', False)
        self.smoothing_time = params.get('smoothing_time', 10)
        self.tree_params['log_level'] = 'info'
        self.graph = ig.Graph(directed=False)
        self.preloaded = False
        self.cbirrt2 = CBiRRT2(self.robot, self.tree_state_space,
                          self.svc, self.interp_fn, params=self.tree_params)
        self.log =  logger if logger is not None else Logger(name='CPRM', handlers=['logging'], level=params.get('log_level', 'debug'))
        self.log.info("CPRM Params: N: {}, k: {}, r: {}".format(
            self.n_samples, self.k, self.ball_radius))

    def preload(self, samples, graph):
        self.graph = graph
        for sample in samples:
            self.graph.vs[utils.val2idx(self.graph, sample)]['value'] = np.array(sample)
        self.samples = samples
        self.preloaded = True

    def plan(self, q_start, q_goal):
        if not self.svc.validate(q_start) and not self.svc.validate(q_goal):
            raise Exception("Starting and ending for planning is invalid according to state validity checker.)")
        if not self.svc.validate(q_start):
            raise Exception("Starting point for planning is invalid according to state validity checker.")
        if not self.svc.validate(q_goal):
                    raise Exception("Ending point for planning is invalid according to state validity checker.)")
        self.generate_roadmap(q_start, q_goal)
        self.log.debug("Attaching start and end to graph...")
        self._attach_start_and_end()
        self.log.debug("Finding feasible best path in graph if available...")
        self.log.debug(self._get_graph_path())
        vertex_sequence = self._get_graph_path()
        if self.smooth_path:
            self.log.debug("Smoothing path...")
            vertex_sequence, path = self._smooth_path(vertex_sequence, self.smoothing_time)
            return path
        else:
            return [self.graph.vs[idx]['value'] for idx in self._get_graph_path()]
    
    def generate_roadmap(self, q_start, q_goal):
        # Initial sampling of roadmap and NN data structure.
        if self.preloaded is False:
            self.log.debug("Generating valid random samples...")
            if self.n_samples <= 100:
                self.samples = self._generate_samples()
            else:
                self.samples = self._generate_samples_parallel()
            self.log.debug("Initializing roadmap with start and goal...")
            self._init_roadmap(q_start, q_goal)
            self.log.debug(len(self.graph.vs))
        else:
            self.log.debug("Using provided samples...")
            self.log.debug("Initializing roadmap with start and goal...")
            self._init_roadmap(q_start, q_goal)
        # print("Attaching start and end to graph...")
        # # self._attach_start_and_end()
        # Create NN datastructure
        self.log.debug("Creating NN datastructure...")
        self.nn = NearestNeighbors(X=np.array(
            self.samples), model_kwargs={"leaf_size": 100})
        # Generate NN connectivity.
        if self.preloaded is False:
            self.log.debug("Adding connections")
            if self.n_samples <= 100:
                self._generate_connections(samples=self.samples)
            else:
                self._generate_connections_parallel(samples=self.samples)
            self.samples = [vs['value'] for vs in self.graph.vs]
            self.log.debug("Recreating NN datastructure from connectable samples...")
            self.nn = NearestNeighbors(X=np.array(
            self.samples), model_kwargs={"leaf_size": 100})
        else:
            self.log.debug("Using provided graph...")

        # self.log.debug("Attaching start and end to graph...")
        # self._attach_start_and_end()
        # self.log.debug("Finding feasible best path in graph if available...")

    def get_path(self, plan):
        points = [self.graph.vs[idx]['value'] for idx in plan]
        pairs = list(zip(points, points[1:]))
        segments = [self.interp_fn(np.array(p[0]), np.array(p[1]))
                    for p in pairs]
        segments = [[list(val) for val in seg] for seg in segments]
        path = []
        for seg in segments:
            path = path + seg
        return path

    def _smooth_path(self, vertex_sequence, smoothing_time=10):
        # create empty tree. 
        smoothing_tree = ig.Graph(directed=True)
        smoothing_tree['name'] = 'smoothing'
        start_time = time.time()

        number_of_shortcuts = 0
        while True:
            current_time = time.time()
            elapsed_time = current_time - start_time

            if elapsed_time > smoothing_time:
                self.log.debug("Finished smoothing iterations in: " + str(int(elapsed_time))  + " seconds")
                break
            # Get two random indeces from path
            center_idx = random.sample(range(0, len(vertex_sequence)), 1)[0]
            lower_range = center_idx - 10 if center_idx - 10 >= 0 else 0
            upper_range = center_idx + 10 if center_idx + 10 <= len(vertex_sequence) else len(vertex_sequence)
            vertex_window = vertex_sequence[lower_range:upper_range]
            rand_idx1, rand_idx2 = random.sample(vertex_window, 2)
            if vertex_sequence.index(rand_idx1) > vertex_sequence.index(rand_idx2):
                continue
            q_old = self.graph.vs[rand_idx1]['value']
            q_s = self.graph.vs[rand_idx2]['value']
            # add points into tree
            # self._add_vertex(smoothing_tree, q_old)
            # self._add_vertex(smoothing_tree, q_s)
            # q_old_name = self._val2str(q_old)
            # q_old_idx = utils.name2idx(smoothing_tree, q_old_name)
            # q_s_name = self._val2str(q_s)
            # q_s_idx = utils.name2idx(smoothing_tree, q_s_name)
            # constrain extended.
            success, smoothed_path_values, _ = self._cbirrt2_connect(q_old, q_s,  add_points_to_samples=False, update_graph=False)
            if success:
                # smoothed_path_values = [smoothing_tree.vs[idx] for idx in self._extract_graph_path(smoothing_tree, q_old_idx, q_s_idx)]
                curr_path_values = [self.graph.vs[idx]['value'] for idx in self._get_graph_path(rand_idx1, rand_idx2)]
                smoothed_path_value_pairs = [(smoothed_path_values[i], smoothed_path_values[(i + 1) % len(smoothed_path_values)]) for i in range(len(smoothed_path_values))][:-1]
                curr_path_values_pairs = [(curr_path_values[i], curr_path_values[(i + 1) % len(curr_path_values)]) for i in range(len(curr_path_values))][:-1]
                smooth_path_distance = sum([self._distance(pair[0], pair[1]) for pair in smoothed_path_value_pairs])
                curr_path_distance = sum([self._distance(pair[0], pair[1]) for pair in curr_path_values_pairs])

                # if the newly found path between indices is shorter, lets use it and add it do the graph
                if smooth_path_distance < curr_path_distance:
                    number_of_shortcuts += 1
                    for q in smoothed_path_values:
                        self._add_vertex(self.graph, q)
                    for pair in smoothed_path_value_pairs:
                        self._add_edge(self.graph, pair[0], pair[1], self._distance(pair[0], pair[1]))
        self.log.debug("Number of shortcuts made during smoothing: {}".format(number_of_shortcuts))
        new_best_vertex_sequence = self._get_graph_path()
        return [self.graph.vs['id'].index(_id) for _id in new_best_vertex_sequence], [self.graph.vs[_id]['value'] for _id in new_best_vertex_sequence]

    def _get_graph_path(self, from_idx=None, to_idx=None):
        if from_idx is None or to_idx is None:
            from_idx = utils.name2idx(self.graph, self.start_name)
            to_idx = utils.name2idx(self.graph, self.goal_name)
        if 'weight' in self.graph.es.attributes():
            return self.graph.get_shortest_paths(from_idx, to_idx, weights='weight', mode='ALL')[0]
        else:
            return self.graph.get_shortest_paths(from_idx, to_idx, mode='ALL')[0]

    def _init_roadmap(self, q_start, q_goal):
        """
        Adds the initial vertices of the start and goal configurations.

        Args:
            q_start (array-like): The starting configuration.
            q_goal (array-like): The goal configuration.
        """
      
        # check if start and goal exist in graph already
        if utils.val2idx(self.graph, q_start) is None:
            self.start_name = utils.val2str(q_start)
            self.graph.add_vertex(self.start_name, **{'value': list(q_start)})
        else:
            self.start_name = utils.val2str(q_start)
            self.graph.vs[utils.val2idx(self.graph, q_start)]['value'] = list(q_start)
        if utils.val2idx(self.graph, q_goal) is None:
            self.goal_name = utils.val2str(q_goal)
            self.graph.add_vertex(self.goal_name, **{'value': list(q_goal)})
        else:
            self.goal_name = utils.val2str(q_goal)
            self.graph.vs[utils.val2idx(self.graph, q_goal)]['value'] = list(q_goal)

    def _attach_start_and_end(self):
        q_start = self.graph.vs[utils.name2idx(self.graph, self.start_name)]['value']
        q_end = self.graph.vs[utils.name2idx(self.graph, self.goal_name)]['value']
        start_added = False
        end_added = False
        for q_near in self._neighbors(q_start, k_override=15, within_ball=False):
            if utils.val2str(q_near) in self.graph.vs['name']:
                if utils.val2idx(self.graph, q_near) != 0:
                    successful, _, _ = self._cbirrt2_connect(q_start, q_near, add_points_to_samples=True)
                    if successful:
                        start_added = True
        for q_near in self._neighbors(q_end, k_override=15, within_ball=False):
            if utils.val2str(q_near) in self.graph.vs['name']:
                if utils.val2idx(self.graph, q_near) != 1:
                    successful, _, _ = self._cbirrt2_connect(q_near, q_end, add_points_to_samples=True)
                    if successful:
                        end_added = True
        if not start_added or not end_added:
            raise Exception("Planning failure! Could not add either start {} and end {} successfully to graph.".format(
                {start_added}, {end_added}))

    def _generate_samples(self):
        # sampling_times = [0]
        count = 0
        valid_samples = []
        while count <= self.n_samples:
            # start_time = timer()
            q_rand = self._sample()
            if np.any(q_rand):
                if self._validate(q_rand):
                    if count % 100 == 0:
                        self.log.debug("{} valid samples...".format(count))
                    valid_samples.append(q_rand)
                    count += 1
                    # sampling_times.append(timer() - start_time)
            # print(sum(sampling_times) / len(sampling_times))
        return valid_samples

    def _generate_samples_parallel(self):
        num_workers = mp.cpu_count()
        samples_per_worker = int(self.n_samples / num_workers)
        worker_fn = partial(
            parallel_projection_worker, sim_context_cls=self.sim_context, sim_config=self.sim_config)
        with mp.get_context("spawn").Pool(num_workers) as p:
            results = p.map(worker_fn, [samples_per_worker] * num_workers)
            return list(itertools.chain.from_iterable(results))

    def _generate_connections(self, samples):
        for q_rand in samples:
            for q_near in self._neighbors(q_rand):
                _, _, _ = self._cbirrt2_connect(q_rand, q_near, add_points_to_samples=False, update_graph=True)

    def _generate_connections_parallel(self, samples):
        evaluated_name_pairs = {}
        point_pairs = []
        for q_rand in samples:
            for q_neighbor in self._neighbors(q_rand):
                if not evaluated_name_pairs.get((utils.val2str(q_rand), utils.val2str(q_neighbor)), False) and not evaluated_name_pairs.get((utils.val2str(q_neighbor), utils.val2str(q_rand)), False):
                    point_pairs.append((q_rand, q_neighbor))
                    evaluated_name_pairs[(utils.val2str(q_rand), utils.val2str(q_neighbor))] = True
                    evaluated_name_pairs[(utils.val2str(q_neighbor), utils.val2str(q_rand))] = True
        self.log.debug("Attempting to connect {} node pairs...".format(len(point_pairs)))
        num_workers = mp.cpu_count()
        batches = np.array_split(point_pairs, num_workers)
        worker_fn = partial(
            parallel_cbirrt_worker, sim_context_cls=self.sim_context, sim_config=self.sim_config, tsr=self.tsr, tree_state_space=self.tree_state_space, interp_fn=self.interp_fn, tree_params=self.tree_params)
        with mp.get_context("spawn").Pool(num_workers) as p:
            batch_results = p.map(worker_fn, batches)
            # for each set of result, we have a dictionary indexed by the evaluated_name_pairs.
            # the value of each dictionary is a dictionary with keys 'points' and 'edges'
            # the points are the actual points and the edges are the edges between points needed between points.
            # we have to add the points to the graphs, then created edges betweenthem.
            self.log.debug("Paralell connections finished, adding points")
            results = {}
            for result in batch_results:
                for named_value_tuple, value in result.items():
                    results[named_value_tuple] = value
            valid_values = []
            valid_edges = []
            valid_edge_weights = []
            for named_tuple in evaluated_name_pairs:
                connection = results.get(named_tuple, None)
                if connection is not None:
                    for point in connection['points']:
                        valid_values.append(list(point))
                    for edge in connection['edges']:
                        valid_edges.append((edge[0], edge[1]))
                        valid_edge_weights.append(self._distance(edge[0], edge[1]))

            values = [self.graph.vs.find(name=self.start_name)["value"],
                  self.graph.vs.find(name=self.goal_name)["value"]] + valid_values
            str_values = [utils.val2str(list(value)) for value in values]
            values = [list(value) for value in values]
            self.graph.add_vertices(len(valid_values))
            self.graph.vs["name"] = str_values
            self.graph.vs["value"] = values
            self.graph.vs['id'] = list(range(0, len(values)))
            # This below step is BY FAR the slowest part of the the graph building since it has to do lookups by attribute which is quite slow for python igrpah.
            edges = [(utils.name2idx(self.graph, utils.val2str(c[0])), utils.name2idx(self.graph, utils.val2str(c[1])))
                    for c in valid_edges]
            self.graph.add_edges(edges)
            self.graph.es['weight'] = valid_edge_weights
            for edge in self.graph.es:
                idx = edge.index
                self.graph.es[idx]['id'] = idx


    def _cbirrt2_connect(self, q_start, q_target, add_points_to_samples=False, update_graph=True):
        self.cbirrt2.reset_planner()

        centroid = (np.array(q_start) + np.array(q_target)) / 2
        radius = self._distance(np.array(q_start), np.array(q_target)) / 2
        
        def _random_config(self):
            return self.state_space.sample(centroid=centroid, radius=radius)

        self.cbirrt2._random_config  = types.MethodType(_random_config, self.cbirrt2)

        graph_plan = self.cbirrt2.plan(self.tsr, q_start, q_target)
        if graph_plan is not None:
            points = [self.cbirrt2.connected_tree.vs[idx]['value']
                      for idx in graph_plan]
            edges = list(zip(points, points[1:]))
            if update_graph:
                for point in points:
                    self._add_vertex(self.graph, point)
                    if add_points_to_samples:
                        if utils.val2str(point) != self.start_name and utils.val2str(point) != self.goal_name:
                            self.samples.append(point)
                for e in edges:
                    e = [np.array(val) for val in e]
                    self._add_edge(self.graph, e[0],
                                e[1], self._distance(e[0], e[1]))
            return True, points, edges
        else:
            return False, [], []

    def _sample(self):
        return np.array(self.state_space.sample())

    def _validate(self, sample):
        return self.svc.validate(sample)

    def _neighbors(self, sample, k_override=None, within_ball=True):
        if k_override is not None:
            k = k_override
        else:
            k = self.k
        distances, neighbors = self.nn.query(sample, k=k)
        if within_ball:
            return [neighbor for distance, neighbor in zip(
                distances, neighbors) if distance <= self.ball_radius and distance > 0]
        else:
            return [neighbor for distance, neighbor in sorted(
                list(zip(distances, neighbors)), key=lambda x: x[0]) if distance > 0]

    def _add_vertex(self, graph, q):
        if not utils.val2str(q) in graph.vs['name']:
            if utils.val2str(q) != self.start_name and utils.val2str(q) != self.goal_name:
                graph.add_vertex(utils.val2str(q), **{'value': list(q)})
                graph.vs[utils.val2idx(graph, q)]['id'] = graph.vs[utils.val2idx(graph, q)].index

    def _add_edge(self, tree, q_from, q_to, weight):
        if utils.val2str(q_from) == self.start_name:
            q_from_idx = utils.name2idx(tree, self.start_name)
        elif utils.val2str(q_from) == self.goal_name:
            q_from_idx = utils.name2idx(tree, self.goal_name)
        else:
            q_from_idx = utils.val2idx(tree, q_from)
        if utils.val2str(q_to) == self.start_name:
            q_to_idx = utils.name2idx(tree, self.start_name)
        elif utils.val2str(q_to) == self.goal_name:
            q_to_idx = utils.name2idx(tree, self.goal_name)
        else:
            q_to_idx = utils.val2idx(tree, q_to)
        if tuple(sorted([q_from_idx, q_to_idx])) not in set([tuple(sorted(edge.tuple)) for edge in tree.es]):
            tree.add_edge(q_from_idx, q_to_idx, **{'weight': weight})

    def _delete_vertex(self, graph, q):
        graph.delete_vertices(utils.val2str(q))

    def _val2idx(self, graph, value):
        return utils.name2idx(graph, utils.val2str(value))

    def _name2idx(self, graph, name):
        return graph.vs.find(name).index

    def _val2name(self, value, places=4):
        def trunc(number, places=4):
            if not isinstance(places, int):
                raise ValueError("Decimal places must be an integer.")
            if places < 1:
                raise ValueError("Decimal places must be at least 1.")
            # If you want to truncate to 0 decimal places, just do int(number).

            with localcontext() as context:
                context.rounding = ROUND_DOWN
                exponent = Decimal(str(10 ** - places))
                return Decimal(str(number)).quantize(exponent).to_eng_string()
        return str([trunc(num, places) for num in value])

    def _distance(self, q1, q2):
        return np.linalg.norm(np.array(q1) - np.array(q2))
