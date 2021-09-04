"""
Functions to compute the modularity of an undirected and unweighted unipartite network.
"""

from core import modularity_maximization as mod_max
from rpy2.robjects.packages import importr
from rpy2.robjects import numpy2ri
import igraph
import numpy as np


def modularity(G, return_groups=False):
    """
    Computes the modularity Q of G using the leading eigenvector methdod.
    :param G:
    :param return_groups:
    :return:
    """
    # Make sure G is square matrix and symmetric
    assert G.shape[0] == G.shape[1]
    assert np.all(np.abs(G - G.T) == 0)

    igraph.arpack_options.mxiter = 10000
    G = G.copy()
    np.fill_diagonal(G, 0)
    g = igraph.Graph.Adjacency(G.tolist(), mode=igraph.ADJ_UNDIRECTED)
    vc = g.community_leading_eigenvector()

    Q = vc.recalculate_modularity()
    if not return_groups:
        return Q

    return Q, vc.membership


def randomize_graph_birewire(G):
    """
    Returns a randomized version of the matrix G that has the same column and row marginal sums.
    The script uses the rpy2 package to call the R function implementing the randomization algorithm
    proposed in:
    https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5168876/

    The R function is implemented in the package BiRewire
    https://bioconductor.org/packages/release/bioc/html/BiRewire.html

    :param G:
    :return:
    """
    numpy2ri.activate()
    G_copy = G.copy()
    np.fill_diagonal(G_copy, 0)
    bc = importr('BiRewire')
    G_rand = np.array(bc.birewire_rewire_undirected(G_copy, verbose=False, exact=False))
    np.fill_diagonal(G_rand, 1)

    assert np.alltrue(np.sum(G_rand, axis=1) == np.sum(G, axis=1))
    numpy2ri.deactivate()

    return G_rand


def randomize_graph_connectance(G):
    """
   Returns a randomized version of the matrix G that has the same connectance as G.
   (i.e, column and row marginal sums are not preserved).
   :param G:
   :return:
   """
    assert G.shape[0] == G.shape[1]
    assert np.all(np.abs(G - G.T) == 0)

    N = G.shape[0]
    num_links = int((np.sum(G) - N) / 2)
    edge_list = [1] * num_links + [0] * int((((N**2) - N) / 2) - num_links)
    np.random.shuffle(edge_list)

    G_rand = np.zeros(G.shape, dtype=int)

    index = 0
    for i in range(N):
        for j in range(i+1, N):
            G_rand[i,j] = G_rand[j,i] = edge_list[index]
            index += 1

    np.fill_diagonal(G_rand, 1)
    return G_rand


def sort_matrix_by_modules(G, modularity_function=None, row_groups=None, col_groups=None, return_groups=False):
    """
    Given an adjacency matrix G, the function sorts the columns (and rows) according to the
    modules detected using the provided modularity_function.
    :param G:
    :param modularity_function:
    :param row_groups:
    :param col_groups:
    :param return_groups:
    :return:
    """
    if row_groups is None or col_groups is None:
        Q_stats = modularity_function(G)
        print(Q_stats)
        row_groups = Q_stats['groups']
        col_groups = Q_stats['groups']

    row_in_group_degrees = np.array([0] * G.shape[0])
    col_in_group_degrees = np.array([0] * G.shape[1])

    row_group_sizes = np.array([0] * G.shape[0])
    col_group_sizes = np.array([0] * G.shape[1])

    row_group_ids, row_group_cnt = np.unique(row_groups, return_counts=True)

    for i, row_group in enumerate(row_group_ids):
        for row in range(G.shape[0]):
            if row_groups[row] == row_group:
                row_group_sizes[row] = row_group_cnt[i]
        for col in range(G.shape[1]):
            if col_groups[col] == row_group:
                col_group_sizes[col] = row_group_cnt[i]

    for group in np.unique(row_groups):

        rows_in_group = [int(row == group) for row in row_groups]
        cols_in_group = [int(col == group) for col in col_groups]
        degrees = np.zeros(G.shape, dtype=int)

        for i in range(G.shape[0]):
            for j in range(G.shape[1]):
                degrees[i, j] = rows_in_group[i]*cols_in_group[j]*G[i, j]

        row_in_group_degrees += degrees.sum(axis=1)
        col_in_group_degrees += degrees.sum(axis=0)

    G = G.copy()
    row_indices = list(range(len(row_groups)))
    row_indices_groups = [(index, module) for g_size, module, degree, index in list(sorted(zip(row_group_sizes, row_groups, row_in_group_degrees, row_indices), reverse=True))]
    row_indices, row_groups_s = map(list, zip(*row_indices_groups))
    G = G[row_indices,:]
    G = G.T

    col_indices = list(range(len(col_groups)))
    col_indices_groups = [(index, module) for g_size, module, degree, index in sorted(zip(col_group_sizes, col_groups, col_in_group_degrees, col_indices), reverse=True)]
    col_indices, col_groups_s = map(list, zip(*col_indices_groups))
    G = G[col_indices,:]
    G = G.T

    if return_groups:
        return G, row_groups_s, col_groups_s
    return G


def cacl_modularity_unipartite(G, NUM_RANDOMIZATIONS=0, RETURN_NORMALIZED=False, randomization_function=None, **kwargs):
    """
    Computes the modularity of the unipartite network G (G must be a symmetric square matrix containing only 0's and 1's).

    :param G: The network
    :param NUM_RANDOMIZATIONS: Number of randomizations used to compute p-Value and z-Score.
    :param RETURN_NORMALIZED: Whether to calculate normalized modularity
    :param randomization_function: The algorithm used to randomize the network (randomize_graph_birewire | randomize_graph_connectance)
    :param kwargs:
    :return:
    """

    if randomization_function is None:
        randomization_function = randomize_graph_birewire

    Q, groups = modularity(G, return_groups=True)

    Q_rand_samples = []
    for i in range(NUM_RANDOMIZATIONS):
        G_rand = randomization_function(G)
        Q_i = modularity(G_rand)
        Q_rand_samples.append(Q_i)

    if NUM_RANDOMIZATIONS > 0:
        p_value = np.sum((np.array(Q_rand_samples) >= Q).astype(int)) / len(Q_rand_samples)
        z_score = (Q - np.mean(Q_rand_samples)) / np.std(Q_rand_samples)

    res = {}
    res['Q'] = Q
    res['groups'] = list(groups)
    res['p_value'] = None
    res['z_score'] = None
    res['Q_rand_samples'] = None
    res['Q_rand'] = None
    res['Q_max'] = None
    res['Q_norm'] = None
    res['Q_maximizer'] = None

    if NUM_RANDOMIZATIONS > 0:
        res['p_value'] = p_value
        res['z_score'] = z_score
        res['Q_rand_samples'] = Q_rand_samples
        res['Q_rand'] = np.mean(Q_rand_samples)

    if RETURN_NORMALIZED:
        MAX_NO_IMPROVEMENT = 5000 if not 'MODULARITY_NORM_MAX_NO_IMPROVEMENT' in kwargs else kwargs['MODULARITY_NORM_MAX_NO_IMPROVEMENT']
        return_optimizer = False if not 'RETURN_MODULARITY_MAXIMIZER' in kwargs else kwargs['RETURN_MODULARITY_MAXIMIZER']
        swapping_function = mod_max.swap_unipartite if not 'swapping_function' in kwargs else kwargs['swapping_function']

        maximizer_res = mod_max.calc_max_modularity(G, MAX_NO_IMPROVEMENT=MAX_NO_IMPROVEMENT, modularity_function=modularity,
                                           swapping_function=swapping_function, verbose=False, return_optimizer=return_optimizer)

        if return_optimizer:
            Q_max, G_max, optimizer = maximizer_res
            res['Q_maximizer'] = optimizer
        else:
            Q_max, G_max = maximizer_res

        Q_rel = (Q - res['Q_rand']) / (Q_max-res['Q_rand'])
        res['Q_max'] = Q_max
        res['Q_norm'] = Q_rel

    return res
