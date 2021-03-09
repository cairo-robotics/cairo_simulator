from math import inf
import itertools
from functools import partial
import multiprocessing as mp
import random
from timeit import default_timer as timer

import numpy as np
import igraph as ig

from cairo_planning.local.evaluation import subdivision_evaluate
from cairo_planning.local.interpolation import cumulative_distance
from cairo_planning.local.neighbors import NearestNeighbors
from cairo_planning.planners.parallel_workers import parallel_connect_worker, parallel_sample_worker, parallel_projection_worker, parallel_cbirrt_worker
from cairo_planning.planners.tree import CBiRRT2

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
        # Start is always at the 0 index.
        self.graph.vs.find(name=str(list(q_start)))["value"] = list(q_start)

        self.goal_name = str(list(q_goal))
        self.graph.add_vertex("goal")
        # Goal is always at the 1 index.
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

    def __init__(self, sim_context_cls, sim_config, robot, tsr, state_space, tree_state_space, state_validity_checker, interpolation_fn, params, tree_params):
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
        self.graph = ig.Graph()
        print("PRM Params: N: {}, k: {}, r: {}".format(
            self.n_samples, self.k, self.ball_radius))

    def plan(self, q_start, q_goal):
        # Initial sampling of roadmap and NN data structure.
        print("Initializing roadmap...")
        self._init_roadmap(q_start, q_goal)
        print("Generating valid random samples...")
        if self.n_samples <= 100:
            samples = self._generate_samples()
        else:
            samples = self._generate_samples_parallel()
        # Create NN datastructure
        print("Creating NN datastructure...")
        self.nn = NearestNeighbors(X=np.array(
            samples), model_kwargs={"leaf_size": 100})
        # Generate NN connectivity.
        print("Generating nearest neighbor connectivity...")
        if self.n_samples <= 100:
            self._generate_connections(samples=samples)
        else:
            self._generate_connections_parallel(samples=samples)
        print("Adding connections")

        print("Attaching start and end to graph...")
        self._attach_start_and_end()
        print("Finding feasible best path in graph if available...")
        return self._get_graph_path()

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

    def _get_graph_path(self):
        return self.graph.get_shortest_paths(self._name2idx(self.graph, 'start'), self._name2idx(self.graph, 'goal'), weights='weight', mode='ALL')[0]

    def _init_roadmap(self, q_start, q_goal):
        """
        Adds the initial vertices of the start and goal configurations.

        Args:
            q_start (array-like): The starting configuration.
            q_goal (array-like): The goal configuration.
        """
        self.graph.add_vertex("start")
        # Start is always at the 0 index.
        self.graph.vs[0]["value"] = q_start
        self.graph.add_vertex("goal")
        # Goal is always at the 1 index.
        self.graph.vs[1]["value"] = q_goal

    def _attach_start_and_end(self):
        start = self.graph.vs[self._name2idx(self.graph, 'start')]['value']
        end = self.graph.vs[self._name2idx(self.graph, 'goal')]['value']
        start_added = False
        end_added = False
        for q_near in self._neighbors(start, k_override=15, within_ball=False):
            if self._val2name(q_near) in self.graph.vs['name']:
                if self._val2idx(self.graph, q_near) != 0:
                    successful = self._cbirrt2_connect(start, q_near)
                    if successful:
                        start_added = True
        for q_near in self._neighbors(end, k_override=15, within_ball=False):
            if self._val2name(q_near) in self.graph.vs['name']:
                if self._val2idx(self.graph, q_near) != 1:
                    successful = self._cbirrt2_connect(q_near, end)
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
                        print("{} valid samples...".format(count))
                    valid_samples.append(q_rand)
                    count += 1
                    # sampling_times.append(timer() - start_time)
            # print(sum(sampling_times) / len(sampling_times))
        return valid_samples

    def _generate_samples_parallel(self):
        num_workers = mp.cpu_count()
        samples_per_worker = int(self.n_samples / num_workers)
        worker_fn = partial(
            parallel_projection_worker, sim_context_cls=self.sim_context, sim_config=self.sim_config, tsr=self.tsr)
        with mp.get_context("spawn").Pool(num_workers) as p:
            results = p.map(worker_fn, [samples_per_worker] * num_workers)
            return list(itertools.chain.from_iterable(results))

    def _generate_connections_parallel(self, samples):
        evaluated_name_pairs = []
        point_pairs = []
        for q_rand in samples:
            for q_neighbor in self._neighbors(q_rand):
                if (self._val2name(q_rand), self._val2name(q_neighbor)) not in evaluated_name_pairs and (self._val2name(q_neighbor), self._val2name(q_rand)) not in evaluated_name_pairs:
                    point_pairs.append((q_rand, q_neighbor))
                    evaluated_name_pairs.append(
                        (self._val2name(q_rand), self._val2name(q_neighbor)))
                    evaluated_name_pairs.append(
                        (self._val2name(q_neighbor), self._val2name(q_rand)))
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
            results = {}
            for result in batch_results:
                for named_value_tuple, value in result.items():
                    results[named_value_tuple] = value

            for named_tuple in evaluated_name_pairs:
                connection = results.get(named_tuple, None)
                if connection is not None:
                    for point in connection['points']:
                        self._add_vertex(self.graph, point)
                    for edge in connection['edges']:
                        self._add_edge(
                            self.graph, edge[0], edge[1], self._distance(edge[0], edge[1]))

    def _cbirrt2_connect(self, q_near, q_target):
        centroid = (np.array(q_near) + np.array(q_target)) / 2
        radius = self._distance(np.array(q_near), np.array(q_target)) / 2

        def _random_config(self):
            return self.state_space.sample(centroid=centroid, radius=radius)

        # monkey-patch goodness to allow for a special type of sampling via hyperball between, and containing q_near and q_target:
        CBiRRT2._random_config = _random_config

        cbirrt2 = CBiRRT2(self.robot, self.tree_state_space,
                          self.svc, self.interp_fn, params=self.tree_params)

        graph_plan = cbirrt2.plan(self.tsr, q_near, q_target)
        if graph_plan is not None:
            points = [cbirrt2.connected_tree.vs[idx]['value']
                      for idx in graph_plan]
            edges = list(zip(points, points[1:]))
            for point in points:
                self._add_vertex(self.graph, point)
            for e in edges:
                self._add_edge(self.graph, e[0],
                               e[1], self._distance(e[0], e[1]))
            return True
        else:
            return False

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
        start_val2name = self._val2name(self.graph.vs[self._name2idx(graph, 'start')]['value'])
        goal_val2name = self._val2name(self.graph.vs[self._name2idx(graph, 'goal')]['value'])
        if not self._val2name(q) in graph.vs['name']:
            if self._val2name(q) != start_val2name and self._val2name(q) != goal_val2name:
                graph.add_vertex(self._val2name(q), **{'value': q})

    def _add_edge(self, tree, q_from, q_to, weight):
        start_val2name = self._val2name(self.graph.vs[self._name2idx(tree, 'start')]['value'])
        goal_val2name = self._val2name(self.graph.vs[self._name2idx(tree, 'goal')]['value'])
        if self._val2name(q_from) == start_val2name:
            q_from_idx = self._name2idx(tree, 'start')
        elif self._val2name(q_from) == goal_val2name:
            q_from_idx = self._name2idx(tree, 'goal')
        else:
            q_from_idx = self._val2idx(tree, q_from)
        if self._val2name(q_to) == start_val2name:
            q_to_idx = self._name2idx(tree, 'start')
        elif self._val2name(q_to) == goal_val2name:
            q_to_idx = self._name2idx(tree, 'goal')
        else:
            q_to_idx = self._val2idx(tree, q_to)
        if tuple(sorted([q_from_idx, q_to_idx])) not in set([tuple(sorted(edge.tuple)) for edge in tree.es]):
            tree.add_edge(q_from_idx, q_to_idx, **{'weight': weight})

    def _delete_vertex(self, graph, q):
        graph.delete_vertices(self._val2name(q))

    def _val2idx(self, graph, value):
        return self._name2idx(graph, self._val2name(value))

    def _name2idx(self, graph, name):
        return graph.vs.find(name).index

    def _val2name(self, value):
        return str(["{:.4f}".format(val) for val in value])

    def _distance(self, q1, q2):
        return np.linalg.norm(q1 - q2)
