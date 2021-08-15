from math import inf
import itertools
from functools import partial
import multiprocessing as mp
import random
from timeit import default_timer as timer
from decimal import Decimal, localcontext, ROUND_DOWN
import time

import numpy as np
import igraph as ig

from cairo_planning.planners.tree import CBiRRT2
from cairo_planning.local.evaluation import subdivision_evaluate
from cairo_planning.local.interpolation import cumulative_distance
from cairo_planning.local.neighbors import NearestNeighbors
from cairo_planning.planners.parallel_workers import parallel_projection_worker
from cairo_simulator.core.log import Logger


__all__ = ['LazyPRM', 'LazyCPRM']


class LazyPRM():

    def __init__(self, state_space, state_validity_checker, interpolation_fn, params):
        self.preloaded = False
        self.graph = ig.Graph()
        self.state_space = state_space
        self.svc = state_validity_checker
        self.interp_fn = interpolation_fn
        self.n_samples = params.get('n_samples', 4000)
        self.k = params.get('k', 10)
        self.attempts = params.get('planning_attempts', 5)
        self.ball_radius = params.get('ball_radius', .55)
        self.samples = []
        print("N: {}, k: {}, r: {}".format(
            self.n_samples, self.k, self.ball_radius))
    
    def preload(self, samples, graph):
        self.samples = samples
        self.graph = graph
        for sample in self.samples:
            self.graph.vs[self._val2idx(self.graph, sample)]['value'] = np.array(sample)
        self.preloaded = True

    def plan(self, q_start, q_goal):
        # Initial sampling of roadmap and NN data structure.
        if self.preloaded is False:
            print("Generating valid random samples...")
            self.samples = self._generate_samples()
            print("Initializing roadmap...")
            self._init_roadmap(q_start, q_goal)
        else:
            print("Adding start and goal points into existing graph structure..")
            self._init_roadmap(q_start, q_goal)
        # Create NN datastructure
        print("Creating NN datastructure...")
        self.nn = NearestNeighbors(X=np.array(
            self.samples), model_kwargs={"leaf_size": 100})
        # Generate NN connectivity.
        if self.preloaded is False:
            print("Generating nearest neighbor connectivity...")
            self.connections = self._generate_connections(samples=self.samples)
            print("Generating graph from samples and connections...")
            self._build_graph(self.samples, self.connections)
        print("Attaching start and end to graph...")
        self._attach_start_and_end()
        print("Finding feasible best path in graph through lazy evaluation if available...")
        path = self.get_lazy_path()
        if len(path) == 0:
            print("Path not found, performing additional sampling and planning attempts")
            for _ in range(self.attempts):
                path = self.replan()
                if len(path) > 0:
                    print("Plan found: {}, smoothing...".format(path))
                    smoothed_path = self._smooth(path)
                    print("Smoothed path indices {}".format(smoothed_path))
                    return smoothed_path
            print("Path not found after additional sampling and planning attempts...")
            return []
        else:
            print("Plan found: {}, smoothing...".format(path))
            smoothed_path = self._smooth(path)
            print("Smoothed path indices {}".format(smoothed_path))
            return smoothed_path

    def replan(self):
        print("Generating more random samples...")
        new_samples = self._generate_samples()
        self.samples = self.samples + new_samples
        # Create NN datastructure
        print("Creating NN datastructure...")
        self.nn = NearestNeighbors(X=np.array(
            self.samples), model_kwargs={"leaf_size": 100})
        # Generate NN connectivity.
        print("Generating nearest neighbor connectivity...")
        self.connections = self._generate_connections(samples=self.samples)
        print("Generating graph from samples and connections...")
        self._build_graph(self.samples, self.connections)
        print("Attaching start and end to graph...")
        self._attach_start_and_end()
        print("Finding feasible best path in graph through lazy evaluation if available...")
        path = self.get_lazy_path()
        return path

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

    def get_lazy_path(self):

        success = False
        # The successful path we will build by iteratively moving through the best current path and redirecting it as needed according
        # to state validity etc,.
        successful_vertex_sequence = []
        # Find the initial best path in the graph, if it exists.
        current_best_plan = self._get_best_path(
            self.start_name, self.goal_name)
        while not success:
            if current_best_plan is None or len(current_best_plan) == 1:
                return []

            invalid_path = True  # we assume an invalid path
            while invalid_path:
                # indicator that there was an invalid node. For now we say there are no invalid nodes since hope is all nodes are valid.
                invalid_node = False
                # first lets validate points randomly in the path random
                shuffled_plan = random.sample(
                    current_best_plan, len(current_best_plan))
                for point_id in shuffled_plan:
                    point_value = self.graph.vs[self.graph.vs['id'].index(
                        point_id)]['value']
                    if self._validate(point_value):  # validate the point value
                        # If its valid, set the validity to true to bypass future collision/validity checks.
                        self.graph.vs.find(self._val2name(list(point_value)))[
                            'validity'] = True
                    else:
                        # if the point is invalid, then we remove it and associated edges from the graph
                        self._remove_node_and_edges(point_id)
                        invalid_node = True
                if invalid_node:  # if there was an invalid node, we find a new path and keep the loop going.
                    current_best_plan = self._get_best_path(
                        self.start_name, self.goal_name)
                    if current_best_plan is None or len(current_best_plan) == 1:
                        return []
                else:
                    # since there was no invalid path after looping, then this current_best_path is ready for edge checking.
                    invalid_path = False

            # let's focus on edges
            points_traverses = [(_id, self.graph.vs[self.graph.vs['id'].index(_id)]['value'])
                                for _id in current_best_plan]
            for idx, _ in enumerate(points_traverses):
                # we first get the current or 'from' vertex id of wherever we are in the path.
                from_id = points_traverses[idx][0]
                # we first get the current or 'from' value of wherever we are in the path.
                from_value = points_traverses[idx][1]
                # get the 'to' point vertex id
                to_id = points_traverses[idx + 1][0]
                # get the 'to' point value
                to_value = points_traverses[idx + 1][1]
                if to_id in successful_vertex_sequence:
                    continue
                # validate the 'to' point value, This is probably redundant right now.
                if self._validate(to_value):
                    # If its valid, set the validity to true to bypass future collision/validity checks.
                    self.graph.vs.find(self._val2name(list(to_value)))[
                        'validity'] = True
                    # generate an interpolated path between 'from' and 'to'
                    segment = self.interp_fn(
                        np.array(from_value), np.array(to_value))
                    # perform segment evaluation for state validity etc
                    if self.graph.es[self.graph.get_eid(self.graph.vs['id'].index(from_id), self.graph.vs['id'].index(to_id))].attributes().get('validity', False):
                        valid = True
                    else:
                        valid = subdivision_evaluate(
                            self.svc.validate, segment)
                    if valid:
                        # if valid, we add the 'to' vertex id since we know we can reach it
                        successful_vertex_sequence.append(to_id)
                        self.graph.es[self.graph.get_eid(self.graph.vs['id'].index(
                            from_id), self.graph.vs['id'].index(to_id))]['validity'] = True
                        goal_idx = self.graph.vs.find(self.goal_name).index
                        if to_id == goal_idx:
                            success = True  # if we made a path to goal, then planning was a success and we can break out
                            break
                        current_best_plan = self._get_best_path(
                            self.start_name, self.goal_name)
                    else:
                        self._remove_edge(from_id, to_id)
                        current_best_plan = self._get_best_path(
                            self.start_name, self.goal_name)
                        successful_vertex_sequence = []
                        break  # we break out of looping with for loop to generate a new best path
                else:
                    # if the 'to' point is invalid, then we remove it and associated edges from the graph
                    self._remove_node_and_edges(to_id)
                    successful_vertex_sequence = []
                    current_best_plan = self._get_best_path(
                        self.start_name, self.goal_name)
                    break
        successful_vertex_sequence.insert(
            0, self.graph.vs.find(self.start_name).index)
        return [self.graph.vs['id'].index(_id) for _id in successful_vertex_sequence]

    def _smooth(self, plan):
        def shortcut(plan):
            idx_range = [i for i in range(0, len(plan))]
            indexed_plan = list(zip(idx_range, plan))
            for curr_idx, vid1 in indexed_plan:
                for test_idx, vid2 in indexed_plan[::-1]:
                    p1 = self.graph.vs[vid1]["value"]
                    p2 = self.graph.vs[vid2]["value"]
                    segment = self.interp_fn(
                        np.array(p1), np.array(p2))
                    valid = subdivision_evaluate(self.svc.validate, segment)
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

    def _get_best_path(self, start_name, end_name):
        graph_idxs = self.graph.get_shortest_paths(
            start_name, end_name, weights='weight', mode='OUT', output='vpath')[0]
        best_path = [self.graph.vs[idx]['id'] for idx in graph_idxs]
        if len(best_path) <= 1:
            return None
        else:
            return best_path

    def _remove_node_and_edges(self, vid):
        graph_idx = self.graph.vs['id'].index(vid)
        self.graph.delete_vertices(graph_idx)

    def _remove_edge(self, vid1, vid2):
        graph_idx1 = self.graph.vs['id'].index(vid1)
        graph_idx2 = self.graph.vs['id'].index(vid2)
        self.graph.delete_edges(self.graph.get_eid(
            graph_idx1, graph_idx2, directed=False, error=True))

    def _init_roadmap(self, q_start, q_goal):
        self.samples = [q_start, q_goal] + self.samples
        self.start_name = self._val2name(q_start)
        self.goal_name = self._val2name(q_goal)
        if 'name' not in self.graph.vs.attributes() or self.start_name not in self.graph.vs['name']:
            self.graph.add_vertex(self.start_name)
            self.graph.vs[self._name2idx(self.graph, self.start_name)]["value"] = q_start
            self.graph.vs.find(self.start_name)['id'] = self.graph.vs.find(self.start_name).index
        if 'name' not in self.graph.vs.attributes() or self.goal_name not in self.graph.vs['name']:
            self.graph.add_vertex(self._val2name(q_goal))
            self.graph.vs[self._name2idx(self.graph, self.goal_name)]["value"] = q_goal
            self.graph.vs.find(self.goal_name)['id'] = self.graph.vs.find(self.goal_name).index


    def _generate_samples(self):
        """In the LazyPRM implementation, we do not check for collision / state validity of sampled points until attempting to traverse a path in the graph.

        Returns:
            [array-like]: Array like object of sampled points
        """
        count = 0
        samples = []
        while count < self.n_samples:
            q_rand = self._sample()
            if np.any(q_rand):
                if count % int(self.n_samples / 4) == 0:
                    print("{} valid samples...".format(count))
                samples.append(q_rand)
                count += 1
        return samples

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
        values = [self.graph.vs.find(self.start_name)["value"],
                  self.graph.vs.find(self.goal_name)["value"]] + samples
        str_values = [self._val2name(list(value)) for value in values]
        values = [list(value) for value in values]
        self.graph.add_vertices(len(values))
        self.graph.vs["name"] = str_values
        self.graph.vs["value"] = values
        self.graph.vs['id'] = list(range(0, self.graph.vcount()))
        # This below step is BY FAR the slowest part of the the graph building since it has to do lookups by attribute which is quite slow for python igrpah.
        edges = [(self._name2idx(self.graph, self._val2name(c[0])), self._name2idx(self.graph, self._val2name(c[1])))
                 for c in connections]
        weights = [c[2] for c in connections]
        self.graph.add_edges(edges)
        self.graph.es['weight'] = weights
        for edge in self.graph.es:
            idx = edge.index
            self.graph.es[idx]['id'] = idx

    def _attach_start_and_end(self):
        start_value = self.graph.vs.find(self.start_name)['value']
        end_value = self.graph.vs.find(self.goal_name)['value']
        start_added = False
        end_added = False
        for q_near in self._neighbors(start_value, k_override=30, within_ball=True):
            if self._name2idx(self.graph, self._val2name(q_near)) != 0:
                successful, local_path = self._extend(start_value, q_near)
                if successful:
                    start_added = True
                    self._add_edge_to_graph(
                        start_value, q_near, self._weight(local_path))
        for q_near in self._neighbors(end_value, k_override=30, within_ball=True):
            if self._name2idx(self.graph, self._val2name(q_near)) != 1:
                successful, local_path = self._extend(q_near, end_value)
                if successful:
                    end_added = True
                    self._add_edge_to_graph(
                        q_near, end_value, self._weight(local_path))
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
        # We keep track of whether or not a node is valid.
        if self.graph.vs.find(self._val2name(list(sample))).attributes().get('validity', False) == True:
            return True
        else:
            return self.svc.validate(sample)

    def _extend(self, q_near, q_rand):
        """In the LazyPRM implementation, we do not check for collision / state validity of connected edges between points
        Args:
            q_near (array-lke): closes neigh point to connect to
            q_rand (array-lke): the random point being added to the graph

        Returns:
            [bool, array-like]: Returns the discrete path generated by the inter_fn of the class
        """
        local_path = self.interp_fn(np.array(q_near), np.array(q_rand))
        return True, local_path

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
        q_near_idx = self._name2idx(self.graph, self._val2name(q_near))
        q_sample_idx = self._name2idx(self.graph,
            self._val2name(q_sample))
        if tuple(sorted([q_near_idx, q_sample_idx])) not in set([tuple(sorted(edge.tuple)) for edge in self.graph.es]):
            self.graph.add_edge(q_near_idx, q_sample_idx,
                                **{'weight': edge_weight})

    def _weight(self, local_path):
        return cumulative_distance(local_path)

    def _val2idx(self, graph, value):
        return self._name2idx(graph, self._val2name(value))

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

class LazyCPRM():

    def __init__(self, sim_context_cls, sim_config, robot, tsr, state_space, tree_state_space, state_validity_checker, interpolation_fn, params, tree_params, logger = None):
        self.preloaded = False
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
        self.smooth_path = params.get('smooth_path', False)
        self.tree_params = tree_params
        self.graph = ig.Graph()
        self.samples = []
        self.log =  logger if logger is not None else Logger(handlers=['logging'], level=params.get('log_level', 'info'))
        self.log.debug("PRM Params: N: {}, k: {}, r: {}".format(
            self.n_samples, self.k, self.ball_radius))

    def preload(self, samples, graph):
        self.samples = samples
        self.graph = graph
        for sample in self.samples:
            self.graph.vs[self._val2idx(self.graph, sample)]['value'] = np.array(sample)
        self.preloaded = True

    def plan(self, q_start, q_goal):
        self.generate_roadmap(q_start, q_goal)
        self.log.debug("Finding feasible best path in graph if available...")
        vertex_sequence, path = self.get_lazy_path()
        if self.smooth_path:
            vertex_sequence = self._smooth_path(vertex_sequence)
            self.log.debug(vertex_sequence)
        return path
    
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
            self.log.debug("Generating nearest neighbor connectivity...")
            connections = self._generate_connections(samples=self.samples)
            # Build the graph structure...
            self.log.debug("Building graph")
            self._build_graph(self.samples, connections)
        else:
            self.log.debug("Using provided graph...")

    def get_path(self, plan):
        points = [self.graph.vs[idx]['value'] for idx in plan]
        pairs = list(zip(points, points[1:]))
        segments = [self._cbirrt2_connect(np.array(p[0]), np.array(p[1]))
                    for p in pairs]
        segments = [[list(val) for val in seg] for seg in segments]
        path = []
        for seg in segments:
            path = path + seg
        return path
    
    def get_lazy_path(self):
        path = []
        success = False
        # The successful path we will build by iteratively moving through the best current path and redirecting it as needed according
        # to state validity etc,.
        successful_vertex_sequence = []
        # Find the initial best path in the graph, if it exists.
        current_best_plan = self._get_best_path(
            self.start_name, self.goal_name)
        while not success:
            if current_best_plan is None or len(current_best_plan) == 1:
                return []

            invalid_path = True  # we assume an invalid path
            while invalid_path:
                # indicator that there was an invalid node. For now we say there are no invalid nodes since hope is all nodes are valid.
                invalid_node = False
                # first lets validate points randomly in the path random
                shuffled_plan = random.sample(
                    current_best_plan, len(current_best_plan))
                for point_id in shuffled_plan:
                    point_value = self.graph.vs[self.graph.vs['id'].index(
                        point_id)]['value']
                    if self._validate(point_value):  # validate the point value
                        # If its valid, set the validity to true to bypass future collision/validity checks.
                        self.graph.vs.find(self._val2name(list(point_value)))[
                            'validity'] = True
                    else:
                        # if the point is invalid, then we remove it and associated edges from the graph
                        self.graph.delete_vertices(self.graph.vs['id'].index(
                        point_id))
                        invalid_node = True
                if invalid_node:  # if there was an invalid node, we find a new path and keep the loop going.
                    current_best_plan = self._get_best_path(
                        self.start_name, self.goal_name)
                    if current_best_plan is None or len(current_best_plan) == 1:
                        return []
                else:
                    # since there was no invalid path after looping, then this current_best_path is ready for edge checking.
                    invalid_path = False

            # let's focus on edges
            points_traverses = [(_id, self.graph.vs[self.graph.vs['id'].index(_id)]['value'])
                                for _id in current_best_plan]
            for idx, _ in enumerate(points_traverses):
                # we first get the current or 'from' vertex id of wherever we are in the path.
                from_id = points_traverses[idx][0]
                # we first get the current or 'from' value of wherever we are in the path.
                from_value = points_traverses[idx][1]
                # get the 'to' point vertex id
                to_id = points_traverses[idx + 1][0]
                # get the 'to' point value
                to_value = points_traverses[idx + 1][1]
                if to_id in successful_vertex_sequence:
                    continue
                # validate the 'to' point value, This is probably redundant right now.
                if self._validate(to_value):
                    # If its valid, set the validity to true to bypass future collision/validity checks.
                    self.graph.vs.find(self._val2name(list(to_value)))[
                        'validity'] = True
                    # generate an interpolated path between 'from' and 'to'
                    success, segment = self._cbirrt2_connect(
                        np.array(from_value), np.array(to_value))
                    # perform segment evaluation for state validity etc
                    valid = False
                    if success and self.graph.es[self.graph.get_eid(self.graph.vs['id'].index(from_id), self.graph.vs['id'].index(to_id))].attributes().get('validity', False):
                        valid = True
                    elif success:
                        valid = subdivision_evaluate(
                            self.svc.validate, segment)
                    else:
                        valid = False
                    if valid:
                        # if valid, we add the 'to' vertex id since we know we can reach it
                        successful_vertex_sequence.append(to_id)
                        self.graph.es[self.graph.get_eid(self.graph.vs['id'].index(from_id), self.graph.vs['id'].index(to_id))]['validity'] = True
                        goal_idx = self.graph.vs.find(self.goal_name).index
                        path = path + segment
                        if to_id == goal_idx:
                            success = True  # if we made a path to goal, then planning was a success and we can break out
                            break
                        current_best_plan = self._get_best_path(
                            self.start_name, self.goal_name)
                    else:
                        self._remove_edge(from_id, to_id)
                        current_best_plan = self._get_best_path(
                            self.start_name, self.goal_name)
                        successful_vertex_sequence = []
                        break  # we break out of looping with for loop to generate a new best path
                else:
                    # if the 'to' point is invalid, then we remove it and associated edges from the graph
                    self.graph.delete_vertices(to_id)
                    successful_vertex_sequence = []
                    current_best_plan = self._get_best_path(
                        self.start_name, self.goal_name)
                    break
        successful_vertex_sequence.insert(
            0, self.graph.vs.find(self.start_name).index)
        return [self.graph.vs['id'].index(_id) for _id in successful_vertex_sequence], path
    
    def _smooth_path(self, graph_path, smoothing_time=6):
        self.log.debug(graph_path)
        # create empty tree. 
        smoothing_tree = ig.Graph(directed=True)
        smoothing_tree['name'] = 'smoothing'
        start_time = time.time()


        while True:
            current_time = time.time()
            elapsed_time = current_time - start_time

            if elapsed_time > smoothing_time:
                self.log.debug("Finished iterating in: " + str(int(elapsed_time))  + " seconds")
                break
            # Get two random indeces from path
            rand_idx1, rand_idx2 = random.sample(graph_path, 2)
            if graph_path.index(rand_idx1) > graph_path.index(rand_idx2):
                continue
            q_old = self.graph.vs[rand_idx1]['value']
            q_s = self.graph.vs[rand_idx2]['value']
            # add points into tree
            # self._add_vertex(smoothing_tree, q_old)
            # self._add_vertex(smoothing_tree, q_s)
            # q_old_name = self._val2str(q_old)
            # q_old_idx = self._name2idx(smoothing_tree, q_old_name)
            # q_s_name = self._val2str(q_s)
            # q_s_idx = self._name2idx(smoothing_tree, q_s_name)
            # constrain extended.
            _, smoothed_path_values = self._cbirrt2_connect(q_old, q_s)
            self.log.debug(smoothed_path_values)
            # smoothed_path_values = [smoothing_tree.vs[idx] for idx in self._extract_graph_path(smoothing_tree, q_old_idx, q_s_idx)]
            curr_path_values = [self.graph.vs[idx]['value'] for idx in self._get_graph_path(rand_idx1, rand_idx2)]
            smoothed_path_value_pairs = [(smoothed_path_values[i], smoothed_path_values[(i + 1) % len(smoothed_path_values)]) for i in range(len(smoothed_path_values))][:-1]
            curr_path_values_pairs = [(curr_path_values[i], curr_path_values[(i + 1) % len(curr_path_values)]) for i in range(len(curr_path_values))][:-1]
            smooth_path_distance = sum([self._distance(pair[0], pair[1]) for pair in smoothed_path_value_pairs])
            curr_path_distance = sum([self._distance(pair[0], pair[1]) for pair in curr_path_values_pairs])

            # if the newly found path between indices is shorter, lets use it and add it do the graph
            if smooth_path_distance < curr_path_distance:
                for q in smoothed_path_values:
                    self._add_vertex(self.graph, q)
                for pair in smoothed_path_value_pairs:
                    self._add_edge(self.graph, pair[0], pair[1], self._distance(pair[0], pair[1]))
        
        return self._get_graph_path()

    def _get_best_path(self, start_name, end_name):
        graph_idxs = self.graph.get_shortest_paths(
            start_name, end_name, weights='weight', mode='OUT', output='vpath')[0]
        best_path = [self.graph.vs[idx]['id'] for idx in graph_idxs]
        if len(best_path) <= 1:
            return None
        else:
            return best_path

    def _get_graph_path(self, from_idx=None, to_idx=None):
        if from_idx is None or to_idx is None:
            from_idx = self._name2idx(self.graph, self.start_name)
            to_idx = self._name2idx(self.graph, self.goal_name)
        if 'weight' in self.graph.es.attributes():
            return self.graph.get_shortest_paths(from_idx, to_idx, weights='weight', mode='OUT')[0]
        else:
            return self.graph.get_shortest_paths(from_idx, to_idx, mode='OUT')[0]

    def _init_roadmap(self, q_start, q_goal):
        """
        Adds the initial vertices of the start and goal configurations.

        Args:
            q_start (array-like): The starting configuration.
            q_goal (array-like): The goal configuration.
        """
        self.samples = [q_start, q_goal] + self.samples
        self.start_name = self._val2name(q_start)
        self.goal_name = self._val2name(q_goal)
        if 'name' not in self.graph.vs.attributes() or self.start_name not in self.graph.vs['name']:
            self.graph.add_vertex(self.start_name)
            self.graph.vs[self._name2idx(self.graph, self.start_name)]["value"] = q_start
            self.graph.vs.find(self.start_name)['id'] = self.graph.vs.find(self.start_name).index
        if 'name' not in self.graph.vs.attributes() or self.goal_name not in self.graph.vs['name']:
            self.graph.add_vertex(self._val2name(q_goal))
            self.graph.vs[self._name2idx(self.graph, self.goal_name)]["value"] = q_goal
            self.graph.vs.find(self.goal_name)['id'] = self.graph.vs.find(self.goal_name).index


    def _attach_start_and_end(self):
        start = self.graph.vs[self._name2idx(self.graph, self.start_name)]['value']
        end = self.graph.vs[self._name2idx(self.graph, self.goal_name)]['value']
        self.samples = self.samples + [start, end]
        

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
            parallel_projection_worker, sim_context_cls=self.sim_context, sim_config=self.sim_config, tsr=self.tsr)
        with mp.get_context("spawn").Pool(num_workers) as p:
            results = p.map(worker_fn, [samples_per_worker] * num_workers)
            return list(itertools.chain.from_iterable(results))[0:self.n_samples]

    def _generate_connections(self, samples):
        connections = []
        for q_rand in samples:
            for q_neighbor in self._neighbors(q_rand):
                valid, local_path = self._extend(
                    np.array(q_rand), np.array(q_neighbor))
                if valid:
                    connections.append(
                        [q_neighbor, q_rand, self._weight(local_path)])
        self.log.debug("{} connections out of {} samples".format(
            len(connections), len(samples)))
        return connections

    def _build_graph(self, samples, connections):
        str_values = [self._val2name(list(value)) for value in samples]
        values = [list(value) for value in samples]
        # we add a bunch of new vertices for the sampled points, but we already have the graph initialized with two points (start and goal)
        self.graph.add_vertices(len(values)-2)
        self.graph.vs["name"] = str_values
        self.graph.vs["value"] = values
        self.graph.vs['id'] = [v.index for v in self.graph.vs]
        edges = [(self._val2idx(self.graph, c[0]), self._val2idx(self.graph, c[1]))
                 for c in connections]
        weights = [c[2] for c in connections]
        self.graph.add_edges(edges)
        self.graph.es['weight'] = weights
        for edge in self.graph.es:
            idx = edge.index
            self.graph.es[idx]['id'] = idx


    def _extend(self, q_near, q_rand):
        """In the LazyPRM implementation, we do not check for collision / state validity of connected edges between points
        Args:
            q_near (array-lke): closes neigh point to connect to
            q_rand (array-lke): the random point being added to the graph

        Returns:
            [bool, array-like]: Returns the discrete path generated by the inter_fn of the class
        """
        local_path = self.interp_fn(np.array(q_near), np.array(q_rand))
        return True, local_path


    def _cbirrt2_connect(self, q_near, q_target):
        centroid = (np.array(q_near) + np.array(q_target)) / 2
        radius = self._distance(np.array(q_near), np.array(q_target)) / 2

        def _random_config(self):
            return self.state_space.sample(centroid=centroid, radius=radius)

        # monkey-patch goodness to allow for a special type of sampling via hyperball between, and containing q_near and q_target:
        CBiRRT2._random_config = _random_config

        self.tree_params['smooth_path'] = False
        cbirrt2 = CBiRRT2(self.robot, self.tree_state_space,
                          self.svc, self.interp_fn, params=self.tree_params, logger=self.log)

        graph_plan = cbirrt2.plan(self.tsr, q_near, q_target)
        if graph_plan is not None:
            points = [cbirrt2.connected_tree.vs[idx]['value']
                      for idx in graph_plan]
            return True, points
        else:
            return False, []

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
        start_val2name = self._val2name(self.graph.vs[self._name2idx(graph, self.start_name)]['value'])
        goal_val2name = self._val2name(self.graph.vs[self._name2idx(graph, self.goal_name)]['value'])
        if not self._val2name(q) in graph.vs['name']:
            if self._val2name(q) != start_val2name and self._val2name(q) != goal_val2name:
                graph.add_vertex(self._val2name(q), **{'value': q})
    
    def _remove_edge(self, vidx1, vidx2):
        start_val2name = self._val2name(self.graph.vs[self._name2idx(self.graph, self.start_name)]['value'])
        goal_val2name = self._val2name(self.graph.vs[self._name2idx(self.graph, self.goal_name)]['value'])
        graph_idx1 = self.graph.vs['id'].index(vidx1)
        graph_idx2 = self.graph.vs['id'].index(vidx2)
        if self.graph.vs[graph_idx1]['name'] not in [start_val2name, goal_val2name] and self.graph.vs[graph_idx2]['name'] not in [start_val2name, goal_val2name]:
            self.graph.delete_edges(self.graph.get_eid(
                graph_idx1, graph_idx2, directed=False, error=True))

    def _add_edge(self, graph, q_from, q_to, weight):
        start_val2name = self._val2name(self.graph.vs[self._name2idx(graph, self.start_name)]['value'])
        goal_val2name = self._val2name(self.graph.vs[self._name2idx(graph, self.goal_name)]['value'])
        if self._val2name(q_from) == start_val2name:
            q_from_idx = self._name2idx(graph, self.start_name)
        elif self._val2name(q_from) == goal_val2name:
            q_from_idx = self._name2idx(graph, self.goal_name)
        else:
            q_from_idx = self._val2idx(graph, q_from)
        if self._val2name(q_to) == start_val2name:
            q_to_idx = self._name2idx(graph, self.start_name)
        elif self._val2name(q_to) == goal_val2name:
            q_to_idx = self._name2idx(graph, self.goal_name)
        else:
            q_to_idx = self._val2idx(graph, q_to)
        if tuple(sorted([q_from_idx, q_to_idx])) not in set([tuple(sorted(edge.tuple)) for edge in graph.es]):
            graph.add_edge(q_from_idx, q_to_idx, **{'weight': weight})

    def _delete_vertex(self, graph, value):
        graph.delete_vertices(self._val2name(value))

    def _val2idx(self, graph, value):
        return self._name2idx(graph, self._val2name(value))

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

    def _weight(self, local_path):
        return cumulative_distance(local_path)

    def _distance(self, q1, q2):
        return np.linalg.norm(q1 - q2)
