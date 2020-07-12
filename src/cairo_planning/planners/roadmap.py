from math import inf
from timeit import default_timer as timer

import numpy as np
import igraph as ig

from cairo_planning.local.evaluation import subdivision_evaluate
from cairo_planning.local.interpolation import cumulative_distance
from cairo_planning.local.neighbors import NearestNeighbors

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
        # Build graph connectivity ignoring first two indices that are the start and end.
        print("Connecting graph...")
        s = timer()
        for idx in range(2, len(self.graph.vs)):
            self._connect_sample(idx)
        e = timer()
        print("Time to connect graph: {}".format(e - s))
        print("Attaching start and end to graph...")
        self._attach_start_and_end()
        print("Finding path...")
        if self._success():
            print("Found path")
            return self.best_path()
        else:
            return []

    def best_path(self):
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
        self.nn = NearestNeighbors(X=np.array(
            [q_start, q_goal]), model_kwargs={"leaf_size": 50})
        # seed the roadmap with a few random samples to build enough neighbors
        valid_samples = 0
        while valid_samples <= self.n_samples:
            q_rand = self._sample()
            if self._validate(q_rand):
                if valid_samples % 100 == 0:
                    print("{} valid samples...".format(valid_samples))
                self._add_vertex(q_rand)
                valid_samples += 1
        self.nn.fit()

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
                    self._add_edge_to_graph(start, q_near, local_path)
                    break
        for q_near in self._neighbors(end, within_ball=False):
            if self._idx_of_point(q_near) != 1:
                successful, local_path = self._extend(q_near, end)
                if successful:
                    end_added = True
                    self._add_edge_to_graph(q_near, end, local_path)
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

    def _connect_sample(self, idx):
        q_curr = self.graph.vs[idx]['value']
        neighbors = self._neighbors(q_curr)
        for q_near in neighbors:
            self._connect(q_near, q_curr)

    def _connect(self, q_near, q_curr):
        idx1 = self._idx_of_point(q_near)
        idx2 = self._idx_of_point(q_curr)
        # Only add if not in graph
        if self.graph.get_eid(idx1, idx2, directed=False, error=False) == -1:
            successful, local_path = self._extend(q_near, q_curr)
            if successful:
                self._add_edge_to_graph(q_near, q_curr, local_path)

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
                distances, neighbors) if distance <= self.ball_radius and neighbor != sample]
        else:
            return [neighbor for distance, neighbor in sorted(
                list(zip(distances, neighbors)), key=lambda x: x[0], reverse=True) if neighbor != sample]

    def _sample(self):
        return np.array(self.state_space.sample())

    def _add_vertex(self, sample):
        self.nn.append(sample)
        self.graph.add_vertex(None, **{'value': list(sample)})

    def _add_edge_to_graph(self, q_near, q_sample, local_path):
        self.graph.add_edge(self._idx_of_point(q_near), self._idx_of_point(
            q_sample), **{'weight': self._weight(local_path)})
        # self.graph.add_edge(self._idx_of_point(q_sample), self._idx_of_point(
        #     q_near), **{'weight': self._weight(local_path)})

    def _weight(self, local_path):
        return cumulative_distance(local_path)

    def _idx_of_point(self, point):
        return self.graph.vs['value'].index(list(point))
    
