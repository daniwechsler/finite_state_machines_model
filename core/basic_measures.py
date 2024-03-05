"""
Implementation of functions to calculate basic network measures.
"""

import numpy as np


def calc_connectance(G):
    """
    Connectance of an unipartite network. It is expected, that
    all entries in the diagonal are 1 (because FSM always interact with themselves).
    :param M:
    :return:
    """
    # Assert matrix symmetric
    assert np.allclose(G, G.T, rtol=0, atol=0)
    M_lower_tri = np.tril(G, k=-1)
    return np.sum(M_lower_tri) / ((G.shape[0]*(G.shape[0]-1)) / 2)


def calc_degrees(G):
    """
    The degree of each node in G.
    :param G:
    :return:
    """
    degrees = G.sum(axis=1)

    return degrees


def H(x):
    """
    The entropy of a list of items.
    :param x:
    :return:
    """
    s, cnt = np.unique(x, return_counts=True)
    p_s = cnt / len(x)
    H = -np.sum([p*np.log2(p) for p in p_s])
    return H


def mutual_information(G, return_values=False):
    """
    Calculates the average mutual information of the interactions across all pairs of
    nodes in the network G (given as an adjacency matrix).
    :param G:
    :param return_values:
    :return:
    """
    mut_inf = []
    for i in range(G.shape[0]):
        for j in range(i+1, G.shape[0]):

            a = G[i, :]
            b = G[j, :]
            m = H(a) + H(b) - H(a*2 + b)
            mut_inf.append(m)

    if return_values:
        return np.mean(mut_inf), mut_inf
    return np.mean(mut_inf)
