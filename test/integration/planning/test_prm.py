from functools import partial

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.collections import PatchCollection, LineCollection
from matplotlib.patches import Rectangle

from cairo_planning.geometric.state_space import R2
from cairo_planning.sampling import StateValidityChecker
from cairo_planning.local.interpolation import parametric_lerp
from cairo_planning.planners import PRM


def plot_prm(found_path, vertices, edges):
    fig, ax = plt.subplots()
    x, y = zip(*found_path)
    ax.plot(x, y, zorder=2, color='red',
            linewidth=4, linestyle='--', label='Regular Path')
    ax.scatter(found_path[0][0], found_path[0][1],
               color='green', s=150, zorder=3)
    ax.scatter(found_path[-1][0], found_path[-1]
               [1], color='blue', s=150, zorder=3)

    line_segments = LineCollection(
        edges, colors='gray', linestyle='solid', alpha=.5, zorder=1)
    ax.add_collection(line_segments)

    x, y = zip(*vertices)
    ax.scatter(x, y, zorder=2, color='blue')

    ax.set_xlim([0, 10])
    ax.set_ylim([0, 10])
    b1 = Rectangle((2, 2), 2, 2)
    b2 = Rectangle((6, 1), 2, 9)
    patch_collection = PatchCollection([b1, b2], alpha=0.4)
    ax.add_collection(patch_collection)
    ax.set_title('PRM')
    ax.legend()
    plt.show()


def box_collision(sample, coordinates=[[0, 0], [2, 2]]):
    x_valid = False
    y_valid = False
    if sample[0] > coordinates[0][0] and sample[0] < coordinates[1][0]:
        x_valid = True
    if sample[1] > coordinates[0][1] and sample[1] < coordinates[1][1]:
        y_valid = True
    if x_valid and y_valid:
        return False
    else:
        return True


if __name__ == "__main__":

    #########################
    # State space selection #
    #########################

    # This inherently uses UniformSampler but a different sampling class could be injected.
    r2_space = R2()

    ##############################
    # State Validity Formulation #
    ##############################
    # There is no self-collision for point object in R2
    self_col_fn = None

    # Create a collision function that combines two box_collision functions
    b1 = partial(box_collision, coordinates=[[2, 2], [4, 4]])
    b2 = partial(box_collision, coordinates=[[6, 1], [8, 10]])
    def col_fn(sample): return all([b1(sample), b2(sample)])

    # In this case, we only have a col_fn.
    svc = StateValidityChecker(
        self_col_func=None, col_func=col_fn, validity_funcs=None)

    ############################################
    # Build the PRM and call the plan function #
    ############################################
    # Create the PRM
    interp = partial(parametric_lerp, steps=10)
    prm = PRM(r2_space, svc, interp, params={
              'n_samples': 2000, 'k': 10, 'ball_radius': .45})

    plan = prm.plan(np.array([1, 1]), np.array([9, 9]))
    print(plan)
    if len(plan) == 0:
        print("Planning failed.")
        exit(1)
    ########################################################################
    # Interpolate between each point in found path through graph and plot  #
    #######################################################################
    path = prm.get_path(plan)

    vertices = [prm.graph.vs[idx]['value']
                for idx in range(0, len(prm.graph.vs))]
    edges = [(prm.graph.vs[e[0]]['value'], prm.graph.vs[e[1]]['value'])
             for e in prm.graph.get_edgelist()]
    path = np.array([np.array(point) for point in path])
    plot_prm(path, vertices, edges)
