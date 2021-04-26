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
from cairo_planning.constraints.projection import project_config

__all__ = ['CBiRRT2']


class CBiRRT2():


    def __init__(self, robot, state_space, state_validity_checker, interpolation_fn, params):
        self.forwards_tree = ig.Graph(directed=True)
        self.backwards_tree = ig.Graph(directed=True)
        self.robot = robot
        self.state_space = state_space
        self.svc = state_validity_checker
        self.interp_fn = interpolation_fn
        self.q_step = params.get('q_step', .12)
        self.epsilon = params.get('epsilon', .1)
        self.e_step = params.get('e_step', .25)
        self.iters = params.get('iters', 1000)
       # print("q_step: {}, epsilon: {}, e_step: {}, BiRRT Iters {}".format(self.q_step, self.epsilon, self.e_step, self.iters))
    
    def plan(self, tsr, start_q, goal_q):
        """ Top level plan function for CBiRRT2. Trees are first initialized with start and end points, constrained birrt is executed, and the path is smoothed.

        Args:
            robot (Cairo Planning Manipulator): Need to for embedded IK/FK functionality.
            tsr (TSR): Task Chain Region for the constraints to plan against.
            start_q (array-like): Starting configuration.
            goal_q (array-like): Ending configuration.
        """
       # print("Initializing trees...")
        self._initialize_trees(start_q, goal_q)
        #print("Running Constrained Bidirectional RRT...")
        self.tree = self.cbirrt(tsr)
        if self.tree is not None:
            #print("Extracting path through graph...")
            graph_path = self._extract_graph_path()
            if len(graph_path) == 1:
                return None
            else:
                #print("Graph path found: {}".format(graph_path))
                return graph_path
        # plan = self.get_plan(graph_path)
        # #self._smooth(path)
        # return plan

    def cbirrt(self, tsr):
        iters=0
        continue_to_plan = True
        tree_swp = self._tree_swap_gen()
        a_tree, b_tree = next(tree_swp)
        while continue_to_plan:
            iters += 1
            if iters > self.iters:
                return None
            q_rand = self._random_config()
            qa_near = self._neighbors(a_tree, q_rand)  # closest leaf value to q_rand
            # extend tree at as far as possible to generate qa_reach
            qa_reach = self._constrained_extend(a_tree, tsr, qa_near, q_rand)
            # closest leaf value of B to qa_reach
            qb_near = self._neighbors(b_tree, qa_reach)  
            # now tree B is extended as far as possible to qa_reach
            qb_reach = self._constrained_extend(b_tree, tsr, qb_near, qa_reach)
            # if the qa_reach and qb_reach are equivalent, the trees are connectable. 
            if self._equal(qa_reach, qb_reach):
                # print("Connecting trees...")
                self.connected_tree = self._join_trees(a_tree, qa_reach, b_tree, qb_reach)
                return self.connected_tree
            # otherwise we swap trees and repeat.
            else:
                 a_tree, b_tree = next(tree_swp)
    
    def _constrained_extend(self, tree, tsr, q_near, q_target):
        q_s = np.array(q_near)
        qs_old = np.array(q_near)
        iters = 1
        prior_distance = self._distance(q_target, q_s)
        while True:
            iters += 1
            if iters >= 1000:
                return q_s
            if self._equal(q_target, q_s):
                return q_s
            # we dont bother to keep a new qs that has traversed further away from the target.
            elif self._distance(q_target, q_s) > self._distance(q_target, qs_old):
                return qs_old
            # set the qs_old to the current value then update
            qs_old = q_s
            # What this update step does is it moves qs off the manifold towards q_target. And then this is projected back down onto the manifold.
            q_s = q_s + min([self.q_step, self._distance(q_target, q_s)]) * (q_target - q_s) / self._distance(q_target, q_s)
            # More problem sepcific versions of constrained_extend use constraint value information 
            # constraints = self._get_constraint_values(tree, qs_old) 
            q_s = self._constrain_config(qs_old=qs_old, q_s=q_s, tsr=tsr)
            if q_s is not None:
                # this function will occasionally osscilate between to projection values.
                if self._val2str(q_s) in tree.vs['name']:
                    # If they've already been added, return the current projection value.
                    return q_s
                elif abs(self._distance(q_s, q_target) - prior_distance) < .005:
                    # or if the projection can no longer move closer along manifold
                    return qs_old
                prior_distance = self._distance(q_s, q_target)
                self._add_vertex(tree, q_s)
                if tree['name'] == 'forwards':
                    self._add_edge(tree, qs_old, q_s, self._distance(qs_old, q_s))
                else:
                    self._add_edge(tree, q_s, qs_old, self._distance(q_s, qs_old))
            else:
                return qs_old

    def _constrain_config(self, qs_old, q_s, tsr):
        # these functions can be very problem specific. For now we'll just assume the most very basic form.
        # futre implementations might favor injecting the constrain_config function 
        q_constrained = project_config(self.robot, tsr, q_s=q_s, q_old=qs_old, epsilon=self.epsilon, q_step=self.q_step, e_step=self.e_step, iter_count=10000)
        if q_constrained is None:
            return None
        elif self.svc.validate(q_constrained):
            return q_constrained
        else:
            return None

    def _extract_graph_path(self,):
        return self.tree.get_shortest_paths(self._name2idx(self.tree, self.start_name), self._name2idx(self.tree, self.goal_name), weights='weight', mode='OUT')[0]

    def get_path(self, plan):
        points = [self.tree.vs[idx]['value'] for idx in plan]
        pairs = list(zip(points, points[1:]))
        segments = [self.interp_fn(np.array(p[0]), np.array(p[1]))
                    for p in pairs]
        segments = [[list(val) for val in seg] for seg in segments]
        path = []
        for seg in segments:
            path = path + seg
        return path

    def _join_trees(self, a_tree, qa_reach, b_tree, qb_reach):

        tree = ig.Graph(directed=True) # a new directed graph
        if a_tree['name'] == 'forwards':
            F = a_tree.copy()
            qf = qa_reach
            B = b_tree.copy()
            qb = qb_reach
        else:
            F = b_tree.copy()
            qf = qb_reach
            B = a_tree.copy()
            qb = qa_reach

        # In certain edge case scenarios, we have two start and goal points very close to each other
        # What this causes is essentially one of the trees to have a length of one.
        # Wile qa_reach and qb_reach are essentially equal according to the distance
        # threshold, it is not necessarily equal to the added start and end nodes
        # hence we get a no such vertex error and potentially our edges are messed up as
        # a result.
        # To fix this, if one of the trees is length 1, we reset the opposite reached point 
        # as the new start/end and  viceversa. If they are both 1 it shouldn't be an issue since
        # they are essnetially the same poitn and we don't need to create an edge between the two
        # points. 
        if len(F.vs) == 1 and len(B.vs) > 1:
            qf_name = self._val2str(qf)
            qf_idx = self._name2idx(F, qf_name)
            qf_value = F.vs[qf_idx]['value']

            qb_name = self._val2str(qb)
            qb_idx = self._name2idx(B, qb_name)
            B.vs[qb_idx]['name'] = qf_name
            B.vs[qb_idx]['value'] = qf_value
        
        if len(F.vs) > 1 and len(B.vs) == 1:
            qb_name = self._val2str(qb)
            qb_idx = self._name2idx(B, qb_name)
            qb_value = F.vs[qb_idx]['value']

            qf_name = self._val2str(qf)
            qf_idx = self._name2idx(F, qf_name)
            F.vs[qf_idx]['name'] = qb_name
            F.vs[qf_idx]['value'] = qb_value
            


        # add all the verticies and edges of the forward tree ot the directed graph
        tree.add_vertices(len(F.vs))
        tree.vs["name"] = F.vs['name']
        tree.vs["value"] = F.vs['value']
    
        F_tree_edges = []
        for e in F.es:
            F_idxs = e.tuple
            F_tree_edges.append((self._name2idx(tree, self._val2str(F.vs[F_idxs[0]]['value'])), self._name2idx(tree, self._val2str(F.vs[F_idxs[1]]['value']))))
        tree.add_edges(F_tree_edges)
        if len(F.es) > 0:
            tree.es['weight'] = F.es['weight']

        # Attach qf to parents/in neighbors of qb, since those parents should be parents to qf
        # Since we built the B graph backwards, we get B's successors
        b_parents = B.successors(self._name2idx(B, self._val2str(qb)))
        # collect the edges and weights of qb to parents in B. We collect by name.
        connection_edges_by_name = []
        connection_edge_weights = []
        qb_name = self._val2str(qb)
        qb_idx = self._name2idx(B, qb_name)
        qf_name = self._val2str(qf)
        for parent in b_parents:
            connection_edges_by_name.append((qf_name, B.vs[parent]['name']))
            connection_edge_weights.append(B.es[B.get_eid(qb_idx,  parent)]['weight'])
        B.delete_vertices(qb_idx)

        if len(B.vs) > 0:
            # add all the verticies and edges of the backwards tree to the directed graph
            curr_names = tree.vs['name'] # we have to snag the current names and values before adding vertices
            curr_values = tree.vs['value']
            tree.add_vertices(len(B.vs))
            tree.vs["name"] = curr_names + list(B.vs['name'])
            tree.vs["value"] = curr_values + list(B.vs['value'])
            B_tree_edges = []
            for e in B.es:
                B_idxs = e.tuple
                B_tree_edges.append((self._name2idx(tree, B.vs[B_idxs[0]]['name']), self._name2idx(tree, self._val2str(B.vs[B_idxs[1]]['value']))))
            if len(B.es) > 0:
                if len(tree.es) > 0:
                    curr_edge_weights = tree.es['weight'] 
                    tree.add_edges(B_tree_edges)
                    tree.es['weight'] = curr_edge_weights + B.es['weight']
                else:
                    tree.add_edges(B_tree_edges)
                    tree.es['weight'] = B.es['weight'] 
        # now add back in the edges from qf to parents of qb in the directed graph/tree
        connection_edges = []
        for edge_name_pair in connection_edges_by_name:
            # get the idx in the tree
            connection_edges.append((self._name2idx(tree, edge_name_pair[0]), self._name2idx(tree, edge_name_pair[1])))
        
        if 'weight' in tree.es.attributes():
            curr_edge_weights = tree.es['weight']
            tree.add_edges(connection_edges)
            tree.es['weight'] = curr_edge_weights + connection_edge_weights
       
        # visual_style = {}
        # visual_style["vertex_color"] =  ["blue" if v['name'] in [self.start_name, self.goal_name] else "white" for v in tree.vs]
        # visual_style["bbox"] = (1200, 1200)

        # visual_style['layout'] = 'tree'
        # ig.plot(tree, **visual_style)
        return tree

    def _neighbors(self, tree, q_s):
        if len(tree.vs) == 1:
            return [v for v in tree.vs][0]['value']
        return sorted([v for v in tree.vs], key= lambda vertex: self._distance(vertex['value'], q_s))[0]['value']

    def _random_config(self):
        # generate a random config to extend towards. This can be biased by whichever StateSpace and/or sampler we'd like to use.
        return np.array(self.state_space.sample())
    
    def _initialize_trees(self, start_q, goal_q):
        self.start_name = self._val2str(start_q)
        self.forwards_tree.add_vertex(self.start_name)
        self.forwards_tree.vs.find(name=self.start_name)['value'] = start_q
        self.forwards_tree['name'] = 'forwards'
       

        self.goal_name = self._val2str(goal_q)
        self.backwards_tree.add_vertex(self.goal_name)
        self.backwards_tree.vs.find(name=self.goal_name)['value'] = goal_q
        self.backwards_tree['name'] = 'backwards'

    def _equal(self, q1, q2):
        if self._distance(q1, q2) <= .05:
            return True
        return False

    def _validate(self, sample):
        return self.svc.validate(sample)
    
    def _tree_swap_gen(self):
        trees = [self.forwards_tree, self.backwards_tree]
        i = 1
        while True:
            idx_1, idx_2 = i%2, (i+1)%2
            yield trees[idx_1], trees[idx_2]
            i += 1

    def _add_vertex(self, tree, q):
        tree.add_vertex(self._val2str(q), **{'value': q})


    def _add_edge(self, tree, q_from, q_to, weight):
        q_from_idx = self._name2idx(tree, self._val2str(q_from))
        q_to_idx = self._name2idx(tree, self._val2str(q_to))
        if self._val2str(q_from) == self.start_name and self._val2str(q_to) == self.goal_name:
            tree.add_edge(q_from_idx, q_to_idx, **{'weight': weight})
        elif tuple(sorted([q_from_idx, q_to_idx])) not in set([tuple(sorted(edge.tuple)) for edge in tree.es]):
            tree.add_edge(q_from_idx, q_to_idx, **{'weight': weight})

    def  _name2idx(self, tree, name):
        try:
            return tree.vs.find(name).index
        except Exception as e:
            print(e)
    
    def _val2str(self, value):
        return str(["{:.8f}".format(val) for val in value])
    
    def _distance(self, q1, q2):
        return np.linalg.norm(np.array(q1) - np.array(q2))