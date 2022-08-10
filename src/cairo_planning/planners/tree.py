from math import inf
import multiprocessing as mp
import random
from timeit import default_timer as timer
import time

import numpy as np
import igraph as ig

from cairo_planning.constraints.projection import project_config
from cairo_planning.geometric.transformation import quat2rpy

from cairo_planning.planners import utils
from cairo_planning.planners.exceptions import MaxItersException, PlanningTimeoutException
from cairo_simulator.core.log import Logger

__all__ = ['CBiRRT2']


class CBiRRT2():


    def __init__(self, robot, state_space, state_validity_checker, interpolation_fn, params, logger=None):
        self.tree = ig.Graph(directed=True)
        self.forwards_tree = ig.Graph(directed=True)
        self.backwards_tree = ig.Graph(directed=True)
        self.robot = robot
        self.state_space = state_space
        self.svc = state_validity_checker
        self.interp_fn = interpolation_fn
        self.smooth_path = params.get('smooth_path', False)
        self.q_step = params.get('q_step', .1)
        self.epsilon = params.get('epsilon', .1)
        self.e_step = params.get('e_step', .25)
        self.iters = params.get('iters', 20000)
        self.max_planning_time = params.get('max_time', 60)
        self.smoothing_time = params.get('smoothing_time', 10)
        self.log =  logger if logger is not None else Logger(name="CBiRRT2", handlers=['logging'], level=params.get('log_level', 'debug'))
        self.log.info("q_step: {}, epsilon: {}, e_step: {}, BiRRT Iters {}".format(self.q_step, self.epsilon, self.e_step, self.iters))
    
    def plan(self, tsr, start_q, goal_q):
        """ Top level plan function for CBiRRT2. Trees are first initialized with start and end points, constrained birrt is executed, and the path is smoothed.

        Args:
            robot (Cairo Planning Manipulator): Need to for embedded IK/FK functionality.
            tsr (TSR): Task Chain Region for the constraints to plan against.
            start_q (array-like): Starting configuration.
            goal_q (array-like): Ending configuration.
        """
        self.log.debug("Initializing trees...")
        self._initialize_trees(start_q, goal_q)
        self.log.debug("Running Constrained Bidirectional RRT...")
        self.tree = self.cbirrt(tsr)
        print("Size of tree: {}".format(len(self.tree.vs)))
        if self.tree is not None:
            self.log.debug("Extracting path through graph...")
            graph_path = self._extract_graph_path()
            self.log.debug("Graph path length prior to smoothing: {}".format(len(graph_path)))
            if len(graph_path) == 1:
                return None
            else:
                if self.smooth_path:
                    self.log.debug("Smoothing for {} seconds".format(self.smoothing_time))
                    self._smooth_path(graph_path, tsr, self.smoothing_time)
                #print("Graph path found: {}".format(graph_path))
                graph_path = self._extract_graph_path()
                self.log.debug("Graph path length after smoothing: {}".format(len(graph_path)))
                return graph_path
        # plan = self.get_plan(graph_path)
        # #self._smooth(path)
        # return plan

    def cbirrt(self, tsr):
        iters=0
        continue_to_plan = True
        tree_swp = self._tree_swap_gen()
        a_tree, b_tree = next(tree_swp)
        tick = time.perf_counter()
        while continue_to_plan:
            iters += 1
            if iters > self.iters:
                self.log.debug("Max iters reach...no feasbile plan.")
                raise MaxItersException("Max CBiRRT2 iterations reached...planning failure.")
            q_rand = self._random_config()
            qa_near = self._neighbors(a_tree, q_rand)  # closest leaf value to q_rand
            # extend tree at as far as possible to generate qa_reach
            qa_reach, _, valid = self._constrained_extend(a_tree, tsr, qa_near, q_rand)
            if not valid:
                continue
            # closest leaf value of B to qa_reach
            qb_near = self._neighbors(b_tree, qa_reach)  
            # now tree B is extended as far as possible to qa_reach
            qb_reach, _, valid = self._constrained_extend(b_tree, tsr, qb_near, qa_reach)
            if not valid:
                continue
            # if the qa_reach and qb_reach are equivalent, the trees are connectable. 
            if self._equal(qa_reach, qb_reach):
                self.connected_tree = self._join_trees(a_tree, qa_reach, b_tree, qb_reach)
                return self.connected_tree
            # otherwise we swap trees and repeat.
            else:
                 a_tree, b_tree = next(tree_swp)
            tock = time.perf_counter()
            if tock - tick > self.timeout_in_seconds:
                raise PlanningTimeoutException()
    
    def reset_planner(self):
        self.tree = ig.Graph(directed=True)
        self.forwards_tree = ig.Graph(directed=True)
        self.backwards_tree = ig.Graph(directed=True)
    
    def _constrained_extend(self, tree, tsr, q_near, q_target):
        generated_values = []
        q_s = np.array(q_near)
        qs_old = np.array(q_near)
        iters = 1
        prior_distance = self._distance(q_target, q_s)
        while True:
            iters += 1
            if iters >= 1000:
                return q_s, generated_values, True
            if self._equal(q_target, q_s):
                return q_s, generated_values, True
            # we dont bother to keep a new qs that has traversed further away from the target.
            elif self._distance(q_target, q_s) > self._distance(q_target, qs_old):
                return qs_old, generated_values, True
            # set the qs_old to the current value then update
            qs_old = q_s
            # What this update step does is it moves qs off the manifold towards q_target. And then this is projected back down onto the manifold.
            q_s = q_s + min([self.q_step, self._distance(q_target, q_s)]) * (q_target - q_s) / self._distance(q_target, q_s)
            # More problem sepcific versions of constrained_extend use constraint value information 
            # constraints = self._get_constraint_values(tree, qs_old)
            
       
            q_s = self._constrain_config(qs_old=qs_old, q_s=q_s, tsr=tsr)
            if q_s is not None:
                # this function will occasionally osscilate between to projection values.
                if utils.val2str(q_s) in tree.vs['name']:
                    # If they've already been added, return the current projection value.
                    return q_s, generated_values, True
                elif abs(self._distance(q_s, q_target) - prior_distance) < .005:
                    # or if the projection can no longer move closer along manifold
                    return qs_old, generated_values, True
                prior_distance = self._distance(q_s, q_target)
                # if q_s is valid AND all of the interpolated points between qs_old and q_s are valid, we add the edge.
                interp = self.interp_fn(qs_old, q_s)
                if self._validate(q_s) and all([self._validate(p) for p in interp]):
                    self._add_vertex(tree, q_s)
                    generated_values.append(q_s)
                    if tree['name'] == 'forwards' or tree['name'] == 'smoothing':
                        self._add_edge(tree, qs_old, q_s, self._distance(qs_old, q_s))
                    else:
                        self._add_edge(tree, q_s, qs_old, self._distance(q_s, qs_old))
                else:
                    return qs_old, generated_values, False
            else:
                # the current q_s is not valid or couldn't be projected so we return the last best value qs_old
                return qs_old, generated_values, False

    def _constrain_config(self, qs_old, q_s, tsr):
        # these functions can be very problem specific. For now we'll just assume the most very basic form.
        # futre implementations might favor injecting the constrain_config function 
        q_constrained = project_config(self.robot, tsr, q_s=q_s, q_old=qs_old, epsilon=self.epsilon, q_step=self.q_step, e_step=self.e_step, iter_count=10000, wrap_to_interval=True)
        if q_constrained is None:
            return None
        if self.svc.validate(q_constrained):
            return q_constrained
        else:
            return None

    def _smooth_path(self, graph_path, tsr, smoothing_time=6):
        # create empty tree. 
        smoothing_tree = ig.Graph(directed=True)
        smoothing_tree['name'] = 'smoothing'
        start_time = time.time()

        if len(graph_path) <= 2:
            return self._extract_graph_path()
 
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
            q_old = self.tree.vs[rand_idx1]['value']
            q_s = self.tree.vs[rand_idx2]['value']
            # add points into tree
            self._add_vertex(smoothing_tree, q_old)
            self._add_vertex(smoothing_tree, q_s)
            q_old_name = utils.val2str(q_old)
            q_old_idx = utils.name2idx(smoothing_tree, q_old_name)
            q_s_name = utils.val2str(q_s)
            q_s_idx = utils.name2idx(smoothing_tree, q_s_name)
            # constrained extend to the potential shortcut point.
            q_reached, added_q_values, valid = self._constrained_extend(smoothing_tree, tsr, q_old, q_s)
            if valid and self._distance(q_reached, q_s) < .01 and len(added_q_values) > 0:
               # since constrain extend does not connect the last point to the target q_s we need to do so.
                self._add_edge(smoothing_tree, added_q_values[-1], q_s, self._distance(added_q_values[-1], q_s))
                smoothed_path_values = [smoothing_tree.vs[idx]['value'] for idx in self._extract_graph_path(smoothing_tree, q_old_idx, q_s_idx)]
                if all([self._validate(p) for p in smoothed_path_values]):
                    curr_path_values = [self.tree.vs[idx]['value'] for idx in self._extract_graph_path(self.tree, rand_idx1, rand_idx2)]
                    smoothed_path_value_pairs = [(smoothed_path_values[i], smoothed_path_values[(i + 1) % len(smoothed_path_values)]) for i in range(len(smoothed_path_values))][:-1]
                    curr_path_values_pairs = [(curr_path_values[i], curr_path_values[(i + 1) % len(curr_path_values)]) for i in range(len(curr_path_values))][:-1]
                    smooth_path_distance = sum([self._distance(pair[0], pair[1]) for pair in smoothed_path_value_pairs])
                    curr_path_distance = sum([self._distance(pair[0], pair[1]) for pair in curr_path_values_pairs])

                    # if the newly found path between indices is shorter, lets use it and add it do the graph
                    if smooth_path_distance < curr_path_distance:
                        # crop off start and end since they already exist and add inbetween vertices of smoothing tree to main
                        for q in smoothed_path_values[1:-1]:
                            self._add_vertex(self.tree, q)
                        for pair in smoothed_path_value_pairs:
                            self._add_edge(self.tree, pair[0], pair[1], self._distance(pair[0], pair[1]))

        return self._extract_graph_path()

    def _extract_graph_path(self, tree=None, from_idx=None, to_idx=None):
        if tree is None:
            tree = self.tree
        if from_idx is None or to_idx is None:
            from_idx = utils.name2idx(tree, self.start_name)
            to_idx = utils.name2idx(tree, self.goal_name)
        if 'weight' in tree.es.attributes():
            return tree.get_shortest_paths(from_idx, to_idx, weights='weight', mode='ALL')[0]
        else:
            return tree.get_shortest_paths(from_idx, to_idx, mode='ALL')[0]

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
        # While qa_reach and qb_reach are essentially equal according to the distance
        # threshold, it is not necessarily equal to the added start and end nodes
        # hence we get a 'no such vertex error' and potentially our edges are messed up as
        # a result.
        # To fix this, if one of the trees is length 1, we reset the opposite reached point 
        # as the new start/end and  viceversa. If they are both 1 it shouldn't be an issue since
        # they are essentially the same point and we don't need to create an edge between the two
        # points. 
        if len(F.vs) == 1 and len(B.vs) > 1:
            qf_name = utils.val2str(qf)
            qf_idx = utils.name2idx(F, qf_name)
            qf_value = F.vs[qf_idx]['value']

            qb_name = utils.val2str(qb)
            qb_idx = utils.name2idx(B, qb_name)
            B.vs[qb_idx]['name'] = qf_name
            B.vs[qb_idx]['value'] = qf_value
            return B
        
        if len(F.vs) > 1 and len(B.vs) == 1:
            qb_name = utils.val2str(qb)
            qb_idx = utils.name2idx(B, qb_name)
            qb_value = F.vs[qb_idx]['value']

            qf_name = utils.val2str(qf)
            qf_idx = utils.name2idx(F, qf_name)
            F.vs[qf_idx]['name'] = qb_name
            F.vs[qf_idx]['value'] = qb_value
            return F
            


        # add all the verticies and edges of the forward tree to the directed graph
        tree.add_vertices(len(F.vs))
        tree.vs["name"] = F.vs['name']
        tree.vs["value"] = F.vs['value']
    
        F_tree_edges = []
        for e in F.es:
            F_idxs = e.tuple
            F_tree_edges.append((utils.name2idx(tree, utils.val2str(F.vs[F_idxs[0]]['value'])), utils.name2idx(tree, utils.val2str(F.vs[F_idxs[1]]['value']))))
        tree.add_edges(F_tree_edges)
        if len(F.es) > 0:
            tree.es['weight'] = F.es['weight']

        # Attach qf to parents/in neighbors of qb, since those parents should be parents to qf
        # Since we built the B graph backwards, we get B's successors
        b_parents = B.successors(utils.name2idx(B, utils.val2str(qb)))
        # collect the edges and weights of qb to parents in B. We collect by name.
        connection_edges_by_name = []
        connection_edge_weights = []
        qb_name = utils.val2str(qb)
        qb_idx = utils.name2idx(B, qb_name)
        qf_name = utils.val2str(qf)
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
                B_tree_edges.append((utils.name2idx(tree, B.vs[B_idxs[0]]['name']), utils.name2idx(tree, utils.val2str(B.vs[B_idxs[1]]['value']))))
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
            connection_edges.append((utils.name2idx(tree, edge_name_pair[0]), utils.name2idx(tree, edge_name_pair[1])))
        
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

    def _neighbors(self, tree, q_s, fraction_random=.1):
        if len(tree.vs) == 1:
            return [v for v in tree.vs][0]['value']
        if random.random() <= fraction_random:
            return random.choice([v for v in tree.vs])['value']
        return sorted([v for v in tree.vs], key= lambda vertex: self._distance(vertex['value'], q_s))[0]['value']

    def _random_config(self):
        # generate a random config to extend towards. This can be biased by whichever StateSpace and/or sampler we'd like to use.
        return np.array(self.state_space.sample())

    def _initialize_trees(self, start_q, goal_q):
        self.start_name = utils.val2str(start_q)
        self.forwards_tree.add_vertex(self.start_name)
        self.forwards_tree.vs.find(name=self.start_name)['value'] = start_q
        self.forwards_tree['name'] = 'forwards'
       

        self.goal_name = utils.val2str(goal_q)
        self.backwards_tree.add_vertex(self.goal_name)
        self.backwards_tree.vs.find(name=self.goal_name)['value'] = goal_q
        self.backwards_tree['name'] = 'backwards'

    def _equal(self, q1, q2):
        if self._distance(q1, q2) <= .05:
            return True
        return False

    def _within_manifold(self, q_s, tsr):
        xyz, quat = self.robot.solve_forward_kinematics(q_s)[0]
        pose = list(xyz) + list(quat2rpy(quat))
        return all(tsr.is_valid(q_s))

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
        tree.add_vertex(utils.val2str(q), **{'value': q})


    def _add_edge(self, tree, q_from, q_to, weight):
        q_from_idx = utils.name2idx(tree, utils.val2str(q_from))
        q_to_idx = utils.name2idx(tree, utils.val2str(q_to))
        if utils.val2str(q_from) == self.start_name and utils.val2str(q_to) == self.goal_name:
            tree.add_edge(q_from_idx, q_to_idx, **{'weight': weight})
        elif tuple(sorted([q_from_idx, q_to_idx])) not in set([tuple(sorted(edge.tuple)) for edge in tree.es]):
            tree.add_edge(q_from_idx, q_to_idx, **{'weight': weight})
    
    def _distance(self, q1, q2):
        return np.linalg.norm(np.array(q1) - np.array(q2))