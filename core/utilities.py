"""
Implementation of utilities that can be used to e.g. print adjacency matrices.
"""

import numpy as np
import time
import copy
import igraph


def compute_G(population, th=1.0, return_cycles=False, return_pins=False):
    """
    Computes the adjacency matrix of the population. If return_cycles and/or return_pins
    are True, the symbols communicate between each pair of FSM on the cylce and before (pin)
    are returned as well.
    :param population:
    :param th:
    :param return_cycles:
    :param return_pins:
    :return:
    """
    G = np.zeros((len(population), len(population)), dtype=int)
    out = []
    pins = []

    for i, fsm_i in enumerate(population):
        out.append([])
        pins.append([])
        for j, fsm_j in enumerate(population):
            G[i, j], out_i, out_j, pin_i, pin_j = fsm_i.encounter(fsm_j, th=th, return_cycle=True, return_pin=True)
            out[i].append("".join(list(out_i)))
            pins[i].append("".join(list(pin_i)))

    if return_cycles:
        if return_pins:
            return G, out, pins
        return G, out

    if return_pins:
        return G, pins, pins
    return G


def print_matrix(G, index_start=1, column_gap=2, P_names=None, A_names=None):
    """
    Prints the adjacency matrix with 0 as not filled squares
    and entries > 0 as filled squares.

    :param G:
    :param w: Where the index of rows and columns start (0 or 1 make sense here)
    :return:
    """

    P, A = np.shape(G)

    if P_names is None:
        P_names = list(range(index_start, P + index_start))

    if A_names is None:
        A_names = list(range(index_start, A + index_start))

    indices = "\t"
    print("\n")

    for j in range(0, A):
        if len(str(j + 1)) == 1:
            indices += str(A_names[j]) + (" " * (column_gap - len(str(A_names[j])) + 1))
        else:
            indices += str(A_names[j]) + (" " * (column_gap - len(str(A_names[j])) + 1))
    print(indices)
    print()
    for i in range(0, P):
        row = str(P_names[i]) + "\t"
        for j in range(0, A):
            if G[i, j] >= 1:
                row += u'\u25A0' + " " * column_gap
            else:
                row += u'\u25A1' + " " * column_gap
        print(row)

    print("\n")


def get_ER_graph(N, L):
    """
    Creates an Erdos-Renyi graph with N nodes an L links with each
    node being connected to itself (G_ii = 1 for all i). The self
    connections are not subtracted form L. Hence for L=0 the only
    connections are those connecting each node to itself.
    :param L:
    :param N:
    :return:
    """

    assert 0 <= L <= ((N**2) - N) / 2

    G_ER = np.zeros((N, N), dtype=int)
    edge_list = [1] * L + [0] * int((((N ** 2) - N) / 2) - L)
    np.random.shuffle(edge_list)

    index = 0
    for i in range(N):
        for j in range(i + 1, N):
            G_ER[i, j] = G_ER[j, i] = edge_list[index]
            index += 1

    np.fill_diagonal(G_ER, 1)
    return G_ER


def get_canonical_network(G):
    """
    Returns a canonical version of the network G (adjacency matrix).
    :param G:
    :return:
    """
    assert set(np.unique(G)) == set([0, 1])
    assert G.shape[0] == G.shape[1]
    assert np.sum(G.T - G) == 0

    g = igraph.Graph.Adjacency(G.tolist(), mode=igraph.ADJ_UNDIRECTED)
    graph_canonical = g.permute_vertices(g.canonical_permutation())
    G_canonical = graph_canonical.get_adjacency()

    return np.array(G_canonical.data)

