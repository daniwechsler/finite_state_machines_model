"""
Greedy algorithm used to find the maximally modular network.
"""
from core.modularity import *
import numpy as np

class ModularityGreedy():
    """
    Minimizes or maximizes the modularity of a given network by swapping connections.
    (connectance of original network is maintained)
    """

    def __init__(self, G, maximize=True, modularity_function=None, swapping_function=None):
        self.G = G
        self.maximize = maximize

        self.modularity_function = modularity_function
        if modularity_function is None:
            self.modularity_function = modularity

        self.swapping_function = swapping_function
        if swapping_function is None:
            self.swapping_function = swap_unipartite

        self.Q_current = self.modularity_function(G)
        self.trace = dict()

    def update(self):

        G_upt = self.G.copy()
        self.swapping_function(G_upt)
        Q_upt = self.modularity_function(G_upt)
        improved = False

        if self.maximize:
            if Q_upt >= self.Q_current: improved = True
        else:
            if Q_upt <= self.Q_current: improved = True

        if improved:
            ret = Q_upt > self.Q_current
            self.G = G_upt
            self.Q_current = Q_upt
            return ret
        return False

    def optimize(self, MAX_NO_IMPROVEMENT=1000, verbose=False):
        no_imporvement = 0
        iteration = 0
        while True:

            if self.update():

                if verbose:
                    print("Q: ", round(self.Q_current, 6), "\t", "Num. steps. without improvement: ", no_imporvement)
                no_imporvement = 0
                self.trace[iteration] = float(self.Q_current)
            else:
                no_imporvement += 1

            iteration += 1
            if no_imporvement >= MAX_NO_IMPROVEMENT:
                self.trace[iteration] = self.Q_current
                break


def swap_unipartite(G):
    """
    Swaps connections in the given network G.
    The method maintains the degree sequence.
    The matrix G must by symmetric and contain only 0's an 1's.

    :param G:
    :return:
    """
    assert G.shape[0] == G.shape[1]
    assert np.all(np.abs(G - G.T) == 0)
    assert np.sum(G) - np.sum(np.diag(G)) > 2

    nodes = list(range(G.shape[1]))

    while True:

        # Chose tow distinct edges at random (a1--a2 and b1--b2)
        np.random.shuffle(nodes)
        a1 = nodes[0]
        a2 = nodes[1]
        b1 = nodes[2]
        b2 = nodes[3]

        if G[a1, a2] == 1 and G[b1, b2] == 1:
            # Determine the allowed rewiring options.
            P = []
            # Check if a1 is not yet connected to b1
            # and a2 not yet connected to b2
            if G[a1, b1] == 0 and G[a2, b2] == 0:
                P.append((b1, b2))
            # Check if a1 is not yet connected to b2
            # and a2 not yet connected to b1
            if G[a1, b2] == 0 and G[a2, b1] == 0:
                P.append((b2, b1))

            # Abort if there is no rewiring possible.
            if len(P) == 0: continue

            # Chose at random one of the (maximally two) possible
            # rewiring option.
            p = P[np.random.randint(len(P))]
            # Do rewiring
            G[a1, a2] = G[a2, a1] = 0
            G[b1, b2] = G[b2, b1] = 0
            G[a1, p[0]] = G[p[0], a1] = 1
            G[a2, p[1]] = G[p[1], a2] = 1
            break


def rewire_unipartite(G):
    """
    The method rewires a single connection of the given network G.
    The method maintains connectance but does not maintain the degree sequence.
    :param G:
    :return:
    """

    assert G.shape[0] == G.shape[1]
    assert np.all(np.abs(G - G.T) == 0)
    assert np.sum(G) - np.sum(np.diag(G)) > 2

    nodes = list(range(G.shape[1]))

    while True:
        # Select two connected nodes at random.
        np.random.shuffle(nodes)
        a1 = nodes[0]
        a2 = nodes[1]

        if G[a1, a2] == 0: continue

        while True:
            # Select two not connected nodes at random.
            remaining_nodes = nodes[2:]
            np.random.shuffle(remaining_nodes)
            a3 = remaining_nodes[0]
            a4 = remaining_nodes[1]

            if G[a3, a4] == 1: continue

            G[a1, a2] = G[a2, a1] = 0
            G[a3, a4] = G[a4, a3] = 1

            break
        break


def calc_max_modularity(G, MAX_NO_IMPROVEMENT=10000, modularity_function=None, swapping_function=None, verbose=False, return_optimizer=False):
    """
    Calculate the maximum modularity of a network of the same size as G.
    If swapping_function=swap_unipartite the set of networks is constrained to those having
    the same degree sequence as G. If If swapping_function=rewire_unipartite the set of networks
    is constrained to those having the same connectance as G.

    :param G: The network
    :param MAX_NO_IMPROVEMENT: The greedy algorithm aborts if this number of iterations no improvement was made.
    :param modularity_function: The modularity function to ues.
    :param swapping_function: The function used to perturb the network.
    :param verbose:
    :param return_optimizer:
    :return:
    """
    G = G.copy()
    mg = ModularityGreedy(G, modularity_function=modularity_function, swapping_function=swapping_function)
    mg.optimize(MAX_NO_IMPROVEMENT=MAX_NO_IMPROVEMENT, verbose=verbose)

    Q_max = mg.Q_current
    if return_optimizer:
        return Q_max, mg.G, mg

    return Q_max, mg.G


