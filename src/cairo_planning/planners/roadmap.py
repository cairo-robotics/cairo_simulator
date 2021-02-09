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
from cairo_planning.planners.parallel_workers import parallel_connect_worker, parallel_sample_worker

__all__ = ['PRM', 'LazyPRM']


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


class LazyPRM():

    def __init__(self, state_space, state_validity_checker, interpolation_fn, params):
        self.graph = ig.Graph()
        self.state_space = state_space
        self.svc = state_validity_checker
        self.interp_fn = interpolation_fn
        self.n_samples = params.get('n_samples', 4000)
        self.k = params.get('k', 5)
        self.attempts = params.get('planning_attempts', 5)
        self.ball_radius = params.get('ball_radius', .55)
        print("N: {}, k: {}, r: {}".format(
            self.n_samples, self.k, self.ball_radius))

    def plan(self, q_start, q_goal):
        # Initial sampling of roadmap and NN data structure.
        print("Initializing roadmap...")
        self._init_roadmap(q_start, q_goal)
        print("Generating valid random samples...")
        self.samples = self._generate_samples()
        # Create NN datastructure
        print("Creating NN datastructure...")
        self.nn = NearestNeighbors(X=np.array(
            self.samples), model_kwargs={"leaf_size": 100})
        # Generate NN connectivity.
        print("Generating nearest neighbor connectivity...")
        connections = self._generate_connections(samples=self.samples)
        print("Generating graph from samples and connections...")
        self._build_graph(self.samples, connections)
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

    # def get_lazy_path(self):
    #     success = False
    #     # The successful path we will build by iteratively moving through the best current path and redirecting it as needed according
    #     # to state validity etc,.
    #     successful_vertex_sequence = []
    #     # Find the initial best path in the graph, if it exists.
    #     current_best_plan = self._get_best_path(self.start_name, self.goal_name)        
    #     while not success:
    #         if current_best_plan is None or len(current_best_plan) == 1:
    #             return []
    #         points = [(_id, self.graph.vs[self.graph.vs['id'].index(_id)]['value'])
    #                   for _id in current_best_plan]
    #         for idx, _ in enumerate(points):
    #             # we first get the current or 'from' vertex id of wherever we are in the path.
    #             from_id = points[idx][0]
    #             # we first get the current or 'from' value of wherever we are in the path.
    #             from_value = points[idx][1]
    #             to_id = points[idx + 1][0]  # get the 'to' point vertex id
    #             to_value = points[idx + 1][1]  # get the 'to' point value
    #             if to_id in successful_vertex_sequence:
    #                 continue
    #             if self._validate(to_value):  # validate the 'to' point value
    #                 # generate an interpolated path between 'from' and 'to'
    #                 segment = self.interp_fn(
    #                     np.array(from_value), np.array(to_value))
    #                 # perform segment evaluation for state validity etc
    #                 valid = subdivision_evaluate(self.svc.validate, segment)
    #                 if valid:
    #                     # if valid, we add the 'to' vertex id since we know we can reach it
    #                     successful_vertex_sequence.append(to_id)
    #                     goal_idx = self.graph.vs.find(self.goal_name).index
    #                     if to_id == goal_idx:
    #                         success = True  # if we made a path to goal, then planning was a success and we can break out
    #                         break
    #                     current_best_plan = self._get_best_path(self.start_name, self.goal_name)
    #                 else:
    #                     self._remove_edge(from_id, to_id)
    #                     current_best_plan = self._get_best_path(self.start_name, self.goal_name)
    #                     successful_vertex_sequence = []
    #                     break  # we break out of looping with for loop to generate a new best path
    #             else:
    #                 # if the 'to' point is invalid, then we remove it and associated edges from the graph
    #                 self._remove_node_and_edges(to_id)
    #                 successful_vertex_sequence = []
    #                 current_best_plan = self._get_best_path(self.start_name, self.goal_name)
    #                 break
    #     successful_vertex_sequence.insert(0, self.graph.vs.find(self.start_name).index)        
    #     return [self.graph.vs['id'].index(_id) for _id in successful_vertex_sequence]
    
    def get_lazy_path(self):

        success = False
        # The successful path we will build by iteratively moving through the best current path and redirecting it as needed according
        # to state validity etc,.
        successful_vertex_sequence = []
        # Find the initial best path in the graph, if it exists.
        current_best_plan = self._get_best_path(self.start_name, self.goal_name)        
        while not success:
            if current_best_plan is None or len(current_best_plan) == 1:
                return []
            
            invalid_path = True # we assume an invalid path
            while invalid_path:
                invalid_node = False # indicator that there was an invalid node. For now we say there are no invalid nodes since hope is all nodes are valid.
                shuffled_plan = random.sample(current_best_plan, len(current_best_plan)) # first lets validate points randomly in the path random
                for point_id in shuffled_plan:
                    point_value = self.graph.vs[self.graph.vs['id'].index(point_id)]['value']
                    if self._validate(point_value):  # validate the point value
                        # If its valid, set the validity to true to bypass future collision/validity checks.
                        self.graph.vs.find(self._value_to_name(list(point_value)))['validity'] = True
                    else:
                        # if the point is invalid, then we remove it and associated edges from the graph
                        self._remove_node_and_edges(point_id)
                        invalid_node = True
                if invalid_node: # if there was an invalid node, we find a new path and keep the loop going.
                    current_best_plan = self._get_best_path(self.start_name, self.goal_name) 
                    if current_best_plan is None or len(current_best_plan) == 1:
                        return [] 
                else:
                    invalid_path = False # since there was no invalid path after looping, then this current_best_path is ready for edge checking.
                        
            # let's focus on edges
            points_traverses = [(_id, self.graph.vs[self.graph.vs['id'].index(_id)]['value'])
                      for _id in current_best_plan]
            for idx, _ in enumerate(points_traverses):
                # we first get the current or 'from' vertex id of wherever we are in the path.
                from_id = points_traverses[idx][0]
                # we first get the current or 'from' value of wherever we are in the path.
                from_value = points_traverses[idx][1]
                to_id = points_traverses[idx + 1][0]  # get the 'to' point vertex id
                to_value = points_traverses[idx + 1][1]  # get the 'to' point value
                if to_id in successful_vertex_sequence:
                    continue
                if self._validate(to_value):  # validate the 'to' point value, This is probably redundant right now.
                    # If its valid, set the validity to true to bypass future collision/validity checks.
                    self.graph.vs.find(self._value_to_name(list(to_value)))['validity'] = True
                    # generate an interpolated path between 'from' and 'to'
                    segment = self.interp_fn(
                        np.array(from_value), np.array(to_value))
                    # perform segment evaluation for state validity etc
                    if self.graph.es[self.graph.get_eid(self.graph.vs['id'].index(from_id), self.graph.vs['id'].index(to_id))].attributes().get('validity', False):
                        valid = True
                    else:
                        valid = subdivision_evaluate(self.svc.validate, segment)
                    if valid:
                        # if valid, we add the 'to' vertex id since we know we can reach it
                        successful_vertex_sequence.append(to_id)
                        self.graph.es[self.graph.get_eid(self.graph.vs['id'].index(from_id), self.graph.vs['id'].index(to_id))]['validity'] = True
                        goal_idx = self.graph.vs.find(self.goal_name).index
                        if to_id == goal_idx:
                            success = True  # if we made a path to goal, then planning was a success and we can break out
                            break
                        current_best_plan = self._get_best_path(self.start_name, self.goal_name)
                    else:
                        self._remove_edge(from_id, to_id)
                        current_best_plan = self._get_best_path(self.start_name, self.goal_name)
                        successful_vertex_sequence = []
                        break  # we break out of looping with for loop to generate a new best path
                else:
                    # if the 'to' point is invalid, then we remove it and associated edges from the graph
                    self._remove_node_and_edges(to_id)
                    successful_vertex_sequence = []
                    current_best_plan = self._get_best_path(self.start_name, self.goal_name)
                    break
        successful_vertex_sequence.insert(0, self.graph.vs.find(self.start_name).index)        
        return [self.graph.vs['id'].index(_id) for _id in successful_vertex_sequence]



    def _smooth(self, plan):
        def shortcut(plan):
            idx_range = [i for i in range(0, len(plan))]
            indexed_plan = list(zip(idx_range, plan))
            for curr_idx, vid1 in indexed_plan:
                for test_idx, vid2 in indexed_plan[::-1]:
                    v1 = self.graph.vs[vid1]
                    v2 = self.graph.vs[vid2]
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
        self.graph.add_vertices(2)
        # Start is always at the 0 index.
        self.graph.vs["name"] = [self._value_to_name(list(q_start)), self._value_to_name(list(q_goal))]
        self.graph.vs["value"] = [list(q_start), list(q_goal)]
        self.start_name, self.goal_name = self._value_to_name(list(q_start)), self._value_to_name(list(q_goal))

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
        str_values = [self._value_to_name(list(value)) for value in values]
        values = [list(value) for value in values]       
        self.graph.add_vertices(len(values))
        self.graph.vs["name"] = str_values
        self.graph.vs["value"] = values
        self.graph.vs['id'] = list(range(0, self.graph.vcount()))
        # This below step is BY FAR the slowest part of the the graph building since it has to do lookups by attribute which is quite slow for python igrpah.
        edges = [(self._idx_of_point(self._value_to_name(c[0])), self._idx_of_point(self._value_to_name(c[1])))
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
            if self._idx_of_point(self._value_to_name(q_near)) != 0:
                successful, local_path = self._extend(start_value, q_near)
                if successful:
                    start_added = True
                    self._add_edge_to_graph(
                        start_value, q_near, self._weight(local_path))
        for q_near in self._neighbors(end_value, k_override=30, within_ball=True):
            if self._idx_of_point(self._value_to_name(q_near)) != 1:
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
        if self.graph.vs.find(self._value_to_name(list(sample))).attributes().get('validity', False) == True:
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
        q_near_idx = self._idx_of_point(self._value_to_name(q_near))
        q_sample_idx = self._idx_of_point(
            self._value_to_name(q_sample))
        if tuple(sorted([q_near_idx, q_sample_idx])) not in set([tuple(sorted(edge.tuple)) for edge in self.graph.es]):
            self.graph.add_edge(q_near_idx, q_sample_idx,
                                **{'weight': edge_weight})

    def _weight(self, local_path):
        return cumulative_distance(local_path)

    def _idx_of_point(self, name):
        return self.graph.vs.find(name).index
    
    def _value_to_name(self, value):
        return str(["{:.4f}".format(val) for val in value])


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
    
