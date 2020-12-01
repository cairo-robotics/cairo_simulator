from math import inf
import itertools
from functools import partial
import multiprocessing as mp
from timeit import default_timer as timer

import numpy as np
import igraph as ig

from cairo_planning.local.evaluation import subdivision_evaluate
from cairo_planning.local.interpolation import cumulative_distance
from cairo_planning.local.neighbors import NearestNeighbors
from cairo_planning.planners.parallel_workers import parallel_connect_worker, parallel_sample_worker

__all__ = ['PRM']


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
        if self._success():
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
        self.graph.add_vertex("start")
        # Start is always at the 0 index.
        self.graph.vs[0]["value"] = list(q_start)
        self.graph.add_vertex("goal")
        # Goal is always at the 1 index.
        self.graph.vs[1]["value"] = list(q_goal)

    def _generate_samples(self):
        count = 0
        valid_samples = []
        while count <= self.n_samples:
            q_rand = self._sample()
            if self._validate(q_rand):
                if count % 500 == 0:
                    print("{} valid samples...".format(count))
                valid_samples.append(q_rand)
                count += 1
        return valid_samples

    def _generate_connections(self, samples):
        connections = []
        for q_rand in samples:
            for q_neighbor in self._neighbors(q_rand):
                valid, local_path = self._extend(
                    np.array(q_neighbor), np.array(q_rand))
                if valid:
                    connections.append(
                        [q_neighbor, q_rand, self._weight(local_path)])
        print("{} connections out of {} samples".format(
                len(connections), len(samples)))
        return connections

    def _build_graph(self, samples, connections):
        values = [self.graph.vs[0]["value"],  self.graph.vs[1]["value"]] + samples
        values = [list(value) for value in values]
        self.graph.add_vertices(len(values))
        self.graph.vs["value"] = values

        edges = [(self._idx_of_point(c[0]), self._idx_of_point(c[1])) for c in connections]
        weights = [c[2] for c in connections]
        self.graph.add_edges(edges)
        self.graph.es['weight'] = weights

    def _attach_start_and_end(self):
        start = self.graph.vs[0]['value']
        end = self.graph.vs[1]['value']
        start_added = False
        end_added = False
        for q_near in self._neighbors(start, within_ball=False):
            if self._idx_of_point(q_near) != 0:
                successful, local_path = self._extend(start, q_near)
                if successful:
                    start_added = True
                    self._add_edge_to_graph(
                        start, q_near, self._weight(local_path))
                    break
        for q_near in self._neighbors(end, within_ball=False):
            if self._idx_of_point(q_near) != 1:
                successful, local_path = self._extend(q_near, end)
                if successful:
                    end_added = True
                    self._add_edge_to_graph(
                        q_near, end, self._weight(local_path))
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

    def _validate(self, sample):
        return self.svc.validate(sample)

    def _extend(self, q_near, q_rand):
        local_path = self.interp_fn(np.array(q_near), np.array(q_rand))
        valid = subdivision_evaluate(self.svc.validate, local_path)
        if valid:
            return True, local_path
        else:
            return False, []

    def _neighbors(self, sample, within_ball=True):
        distances, neighbors = self.nn.query(sample, k=self.k)
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
            self.graph.add_edge(q_near_idx, q_sample_idx, **{'weight': edge_weight})

    def _weight(self, local_path):
        return cumulative_distance(local_path)

    def _idx_of_point(self, point):
        return self.graph.vs['value'].index(list(point))


class PRMParallel():

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
        self.graph.add_vertex("start")
        # Start is always at the 0 index.
        self.graph.vs[0]["value"] = list(q_start)
        self.graph.add_vertex("goal")
        # Goal is always at the 1 index.
        self.graph.vs[1]["value"] = list(q_goal)

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
        values = [self.graph.vs[0]["value"],  self.graph.vs[1]["value"]] + samples
        values = [list(value) for value in values]
        self.graph.add_vertices(len(values))
        self.graph.vs["value"] = values

        edges = [(self._idx_of_point(c[0]), self._idx_of_point(c[1])) for c in connections]
        weights = [c[2] for c in connections]
        self.graph.add_edges(edges)
        self.graph.es['weight'] = weights

    def _attach_start_and_end(self):
        start = self.graph.vs[0]['value']
        end = self.graph.vs[1]['value']
        start_added = False
        end_added = False
        for q_near in self._neighbors(start, within_ball=False):
            if self._idx_of_point(q_near) != 0:
                successful, local_path = self._extend(start, q_near)
                if successful:
                    start_added = True
                    self._add_edge_to_graph(
                        start, q_near, self._weight(local_path))
                    break
        for q_near in self._neighbors(end, within_ball=False):
            if self._idx_of_point(q_near) != 1:
                successful, local_path = self._extend(q_near, end)
                if successful:
                    end_added = True
                    self._add_edge_to_graph(
                        q_near, end, self._weight(local_path))
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
        self.graph.add_edge(q_near_idx, q_sample_idx, **{'weight': edge_weight})

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

    def _idx_of_point(self, point):
        return self.graph.vs['value'].index(list(point))
