"""
Evolves a population of FSM towards a specified value of modularity and connectance.
"""

from core.fsm_minimization import minimize_fsm
from core.modularity import *
from core.modularity_maximization import calc_max_modularity, rewire_unipartite
from core.finite_state_machine import get_random_population
from core.utilities import *
from core.basic_measures import *
import io
import json
import numpy as np
import time
import matplotlib.pyplot as plt


class OptimizerUnipartite(object):

    EPSILON = 0.01

    def __init__(self, fsm, function, th, mu=1.0, REPLACE_PARENT=True):
        self.fsm = fsm
        self.function = function
        self.mu = mu
        self.th = th
        self.done = False
        self.G = compute_G(self.fsm, th=th)
        self.REPLACE_PARENT = REPLACE_PARENT

    def select_fsm(self):
        """
        Returns the index of a randomly chosen fsm
        :return:
        """

        i_reproduce = np.random.randint(0, self.G.shape[0])

        if self.REPLACE_PARENT:
            # If REPLACE_PARENT is set the offspring replaces its parent in the population
            # (and not another randomly chosen fsm).
            return i_reproduce, i_reproduce

        i_kill = np.random.randint(0, self.G.shape[0])

        return i_reproduce, i_kill

    def update(self):

        i_reproduce, i_kill = self.select_fsm()

        offspring = self.fsm[i_reproduce].clone()
        offspring.mutate(self.mu)

        G_new = np.copy(self.G)

        for k, fsm_k in enumerate(self.fsm):
            G_ik = offspring.encounter(fsm_k, self.th)
            G_new[i_kill, k] = G_ik
            G_new[k, i_kill] = G_ik
            if i_kill == k:
                G_new[i_kill, k] = 1

        fsm_list_new = self.fsm[:]
        fsm_list_new[i_kill] = offspring

        accept = self.function(self, G_new, fsm_list_new)

        if accept:
            self.G = G_new
            self.fsm = fsm_list_new
            return True

        return False

    def optimize(self, MAX_ITERATIONS=100000, verbose=False):
        iter = 0
        while True:
            upd = self.update()
            if upd:
                if verbose:
                    self.function.print_state(self.G)

            if self.done:
                return iter

            if iter > MAX_ITERATIONS:
                print("Max. iterations reached.")
                break
            iter += 1
        return iter


class ModularityTargetOptimizer(object):

    def __init__(self, c_target, Q_norm_target, Q_rand, Q_max, Q_norm_epsilon=0.005, c_epsilon=0.01):

        self.Q_rand = Q_rand
        self.Q_max = Q_max
        self.c_target = c_target
        self.Q_norm_target = Q_norm_target

        self.Q_norm_current = None
        self.groups = None
        self.c_current = None

        self.Q_epsilon = Q_norm_epsilon
        self.c_epsilon = c_epsilon

        self.d_c_old = None
        self.d_Q_norm_old = None
        self.G_current = None

    def __call__(self, optimizer, G_new, fsm_list):

        # Compute connectance and modularity of new network
        c_new = calc_connectance(G_new)
        Q_stats = cacl_modularity_unipartite(G_new)
        Q_norm = (Q_stats['Q'] - self.Q_rand) / (self.Q_max - self.Q_rand)

        Q_norm_new = Q_norm
        d_c_new = c_new - self.c_target
        d_Q_norm_new = Q_norm_new - self.Q_norm_target

        # Abort, if both cconnectance and modularity are within tolerance range
        if abs(d_c_new) < self.c_epsilon and abs(d_Q_norm_new) < self.Q_epsilon:
            optimizer.done = True
            self.G_current = G_new
            return True

        if self.d_c_old is None:
            c_initial = calc_connectance(optimizer.G)
            self.d_c_old = c_initial - self.c_target
            self.c_current = c_initial

        if self.d_Q_norm_old is None:

            Q_stats = cacl_modularity_unipartite(G_new)
            Q_norm = (Q_stats['Q'] - self.Q_rand) / (self.Q_max - self.Q_rand)
            Q_norm_initial = Q_norm
            self.d_Q_norm_old = Q_norm_initial - self.Q_norm_target
            self.Q_norm_current = Q_norm_initial
            self.groups = Q_stats['groups']

        if abs(d_c_new) > abs(self.d_c_old) and abs(d_c_new) > self.c_epsilon:
            return False

        if abs(d_Q_norm_new) > abs(self.d_Q_norm_old) and abs(d_Q_norm_new) > self.Q_epsilon:
            return False

        self.d_c_old = d_c_new
        self.d_Q_norm_old = d_Q_norm_new
        self.Q_norm_current = Q_norm_new
        self.c_current = c_new
        self.groups = Q_stats['groups']
        self.G_current = G_new
        return True

    def print_state(self, G):

        printMatrix(sort_matrix_by_modules(G, row_groups=self.groups, col_groups=self.groups))
        print("Q: ", self.Q_norm_current, " c: ", self.c_current)


class MutualInformationTargetOptimizer(object):

    def __init__(self, c_target, mut_target,  mut_epsilon=0.0005, c_epsilon=0.005):

        self.c_target = c_target
        self.mut_target = mut_target

        self.mut_epsilon = mut_epsilon
        self.c_epsilon = c_epsilon

        self.d_c_old = None
        self.d_mut_old = None
        self.c_current = None
        self.mut_current = None
        self.trace = []

    def __call__(self, optimizer, G_new, fsm_list=None, axis=None):

        c_new = calc_connectance(G_new)
        mut_new = mutual_information(G_new)

        d_c_new = c_new - self.c_target
        d_mut_new = mut_new - self.mut_target

        if abs(d_c_new) < self.c_epsilon and abs(d_mut_new) < self.mut_epsilon:
            optimizer.done = True
            return True

        if self.d_c_old is None:
            c_initial = calc_connectance(optimizer.G)
            self.d_c_old = c_initial - self.c_target
            self.c_current = c_initial

        if self.d_mut_old is None:
            mut_initial = mutual_information(G_new)
            self.d_mut_old = mut_initial - self.mut_target
            self.mut_current = mut_initial

        if abs(d_c_new) > abs(self.d_c_old) and abs(d_c_new) > self.c_epsilon:
            return False

        if abs(d_mut_new) > abs(self.d_mut_old) and abs(d_mut_new) > self.mut_epsilon:
            return False

        self.d_c_old = d_c_new
        self.d_mut_old = d_mut_new

        self.c_current = c_new
        self.mut_current = mut_new
        self.trace.append(mut_new)
        return True

    def print_state(self, G):

        mut, mut_ = mutual_information(G, return_values=True)
        print('c: {0:7}   mut:{1:8}   (dmut:{2:8})'.format(round(self.c_current, 5), round(mut, 5), round(self.d_mut_old, 5)))


class ConnectanceTargetOptimizer(object):

    def __init__(self, c_target, c_epsilon=0.02):

        self.c_target = c_target

        self.c_epsilon = c_epsilon

        self.d_c_old = None
        self.c_current = None

    def __call__(self, optimizer, G_new, fsm_list=None, axis=None):

        c_new = calc_connectance(G_new)
        d_c_new = c_new - self.c_target

        if abs(d_c_new) < self.c_epsilon:
            optimizer.done = True
            return True

        if self.d_c_old is None:
            c_initial = calc_connectance(optimizer.G)
            self.d_c_old = c_initial - self.c_target
            self.c_current = c_initial

        if abs(d_c_new) > abs(self.d_c_old) and abs(d_c_new) > self.c_epsilon:
            return False

        self.d_c_old = d_c_new
        self.c_current = c_new

        return True

    def print_state(self, G):

        printMatrix(G)
        print("c: ", self.c_current)


def run_modularity_optimization(n, N, c_target, Q_norm_target, th,
                                MAX_UPDATES = 100000,
                                NUM_RANDOMIZATIONS=500,
                                MAX_MODULARITY_MAX_NO_IMPROVEMENT=10000,
                                FSM_COMPLEXITY_MAX_NO_IMPROVE=10000,
                                DO_RANDOMIZATION_STEP=True,
                                C_EPSILON=0.0025,
                                Q_NORM_EPSILON = 0.001,
                                MUT_EPSILON = 0.0001,
                                NUM_Q_MAX_SAMPLES = 1,
                                plot_stats=False,
                                verbose=False):

    """
    The function evolves a population of N FSM with n states each until that
    the interaction network has connectance c_target and normalized modularity
    Q_norm_target.

    :param n: Number of FSM states
    :param N: Number of FSM
    :param c_target: Target value of connectance
    :param Q_norm_target: Target value of normalized modularity
    :param th: Interaction specificity (delta)
    :param MAX_UPDATES:
    :param NUM_RANDOMIZATIONS:
    :param MAX_MODULARITY_MAX_NO_IMPROVEMENT:
    :param FSM_COMPLEXITY_MAX_NO_IMPROVE:
    :param DO_RANDOMIZATION_STEP: Whether to first evolve towards network with average pairwise mutual information
    of random network (default is True)
    :param C_EPSILON: Tolerance for connectance
    :param Q_NORM_EPSILON: Tolerance for modularity
    :param MUT_EPSILON: Tolerance for mutual information
    :param NUM_Q_MAX_SAMPLES:
    :param plot_stats:
    :param verbose:
    :return:
    """
    mu = 0.1
    #######################################
    # Create the initial community (random)
    #######################################
    community = get_random_population(N, n)

    #######################################
    # Compute E[I], Q_rand, Q_max
    #######################################
    start_time = time.time()
    L = int(c_target * ((N ** 2) - N) / 2)
    Q_rand_samples = []
    I_rand = []

    # Compute expected modularity and expected mutual information of
    # ER networks.
    for i in range(NUM_RANDOMIZATIONS):
        G_rand = get_ER_graph(N, L)
        I_rand.append(mutual_information(G_rand))
        Q_rand_samples.append(modularity(G_rand))

    E_I = np.mean(I_rand)
    Q_rand = np.mean(Q_rand_samples)
    duration_Q_rand = time.time() - start_time

    # Compute expected maximal modularity of ER-network
    start_time = time.time()
    Q_max_ = []
    for i in range(NUM_Q_MAX_SAMPLES):
        G_rand = get_ER_graph(N, L)
        Q_max, G_max = calc_max_modularity(G_rand, MAX_NO_IMPROVEMENT=MAX_MODULARITY_MAX_NO_IMPROVEMENT, modularity_function=modularity,
                                           swapping_function=rewire_unipartite, verbose=verbose,
                                           return_optimizer=False)
        Q_max_.append(Q_max)

    Q_max = np.max(Q_max_)
    duration_Q_max = time.time() - start_time

    #######################################
    # Target mutual information
    #######################################

    if DO_RANDOMIZATION_STEP:
        func_c = MutualInformationTargetOptimizer(c_target, E_I, mut_epsilon=MUT_EPSILON, c_epsilon=C_EPSILON)
    else:
        func_c = ConnectanceTargetOptimizer(c_target, c_epsilon=C_EPSILON)

    # Run optimization to E[I]
    start_time_randomization = time.time()
    opt = OptimizerUnipartite(community, func_c, th=th, mu=mu)
    num_iterations_randomization = opt.optimize(verbose=True, MAX_ITERATIONS=MAX_UPDATES)

    if DO_RANDOMIZATION_STEP:
        I_end = func_c.mut_current
    else:
        I_end = mutual_information(opt.G)

    duration_randomization = time.time() - start_time_randomization
    print("Target Mutual Information Reached [", round(duration_randomization), " s]")

    #######################################
    # Compute stats of irregular network
    #######################################
    G_start = opt.G.copy()
    c_start = calc_connectance(G_start)

    Q_start, groups_start = modularity(G_start, return_groups=True)

    num_groups_start = len(np.unique(groups_start))
    Q_p_value_start = np.sum((np.array(Q_rand_samples) >= Q_start).astype(int)) / len(Q_rand_samples)
    Q_z_score_start = (Q_start - np.mean(Q_rand_samples)) / np.std(Q_rand_samples)
    Q_norm_start = (Q_start - Q_rand) / (Q_max - Q_rand)

    G_sorted, row_indices, col_indices = sort_matrix_by_modules(G_start, row_groups=groups_start, col_groups=groups_start,
                                                                return_groups=True)

    print('{0:16}  {1}'.format('c_start:', round(c_start, 5)))
    printMatrix(G_sorted)

    ###################################
    # Target Modularity
    ###################################
    start_time = time.time()

    func = ModularityTargetOptimizer(c_target, Q_norm_target, Q_rand, Q_max, Q_norm_epsilon=Q_NORM_EPSILON, c_epsilon=C_EPSILON)
    opt = OptimizerUnipartite(opt.fsm, func, th=th, mu=mu)
    num_iterations_modularity = opt.optimize(verbose=verbose, MAX_ITERATIONS=MAX_UPDATES)

    end_time = time.time()
    duration_modulairty = end_time - start_time
    print("Target Modularity reached [", round(duration_modulairty, 2), " s]")


    #######################################
    # Compute states for modular network
    #######################################
    G_end = opt.G.copy()
    c_end = calc_connectance(G_end)

    # Compute average trait complexity
    start_time = time.time()
    rel_complexity_end = minimize_fsm(opt.fsm, th=th,
                                                        MAX_NO_IMPROVEMENT=FSM_COMPLEXITY_MAX_NO_IMPROVE,
                                                        verbose=False)

    duration_complexity = time.time() - start_time

    Q_end, groups_end = modularity(G_end, return_groups=True)
    Q_norm_end = (Q_end - Q_rand) / (Q_max - Q_rand)
    num_groups_end = len(np.unique(groups_end))
    Q_p_value_end = np.sum((np.array(Q_rand_samples) >= Q_end).astype(int)) / len(Q_rand_samples)
    Q_z_score_end = (Q_end - np.mean(Q_rand_samples)) / np.std(Q_rand_samples)

    G_sorted, row_indices, col_indices = sort_matrix_by_modules(G_end, row_groups=groups_end, col_groups=groups_end,
                                                                return_groups=True)
    printMatrix(G_sorted)

    print('{0:16}  {1}'.format('Q_norm_start:', round(Q_norm_start, 3)))
    print('{0:16}  {1}'.format('Q_norm_end:', round(Q_norm_end, 3)))
    print('{0:16}  {1}'.format('Q_norm_target:', round(Q_norm_target, 3)))
    print()
    print('{0:16}  {1}'.format('Q_start:', round(Q_start, 3)))
    print('{0:16}  {1}'.format('Q_end:', round(Q_end, 3)))
    print()
    print('{0:16}  {1}'.format('Q_rand:', round(Q_rand, 3)))
    print('{0:16}  {1}'.format('Q_max:', round(Q_max, 3)))
    print()
    print('{0:16}  {1}'.format('I_end:', round(I_end, 3)))
    print('{0:16}  {1}'.format('E_I:', round(E_I, 3)))

    # Convert start and end network to string representation
    G_start_str = io.BytesIO()
    np.savetxt(G_start_str, G_start, delimiter=" ", fmt='%i')

    G_end_str = io.BytesIO()
    np.savetxt(G_end_str, G_end, delimiter=" ", fmt='%i')

    stats = {
        'c_start': c_start,
        'c_end': c_end,
        'avg_rel_complexity_end': rel_complexity_end,
        'Q_start': Q_start,
        'Q_end': Q_end,
        'Q_rand': Q_rand,
        'Q_norm_start': Q_norm_start,
        'Q_norm_end': Q_norm_end,
        'Q_max': Q_max,
        'Q_p_value_start': Q_p_value_start,
        'Q_z_score_start': Q_z_score_start,
        'Q_p_value_end': Q_p_value_end,
        'Q_z_score_end': Q_z_score_end,
        'E_I': E_I,
        'I_end' : I_end,
        'Q_rand_samples': json.dumps(Q_rand_samples),
        'G_start': G_start_str.getvalue(),
        'G_end': G_end_str.getvalue(),
        'num_groups_start': num_groups_start,
        'num_groups_end': num_groups_end,
        'num_iterations_modularity': num_iterations_modularity,
        'num_iterations_randomization': num_iterations_randomization,
        'duration_randomization': duration_randomization,
        'duration_modulairty': duration_modulairty,
        'duration_complexity': duration_complexity,
        'duration_Q_max':duration_Q_max,
        'duration_Q_rand': duration_Q_rand,

    }

    if plot_stats:
        fig, ax = plt.subplots(nrows=1, ncols=1, sharex=False, sharey=False, squeeze=False, figsize=(5, 4), dpi=120)

        ax[0, 0].hist(Q_rand_samples, color='blue', alpha=0.5, label='swap')
        ax[0, 0].axvline(Q_start, color='red', label='Q_start')
        ax[0, 0].axvline(Q_rand, color='grey', label='Q_rand_start')
        ax[0, 0].axvline(Q_max, color='orange', label='Q_max')
        ax[0, 0].axvline(Q_end, color='blue', label='Q_end', linestyle='--')
        ax[0, 0].legend(fontsize=8)
        plt.show()

    return stats



if __name__ == '__main__':
    import cProfile
    np.random.seed(229)
    n = 20
    N = 30
    c_target = 0.5

    VERBOSE=True
    Q_target = 0.0
    th = 1.0

    #def run ():
    run_modularity_optimization(n, N, c_target, Q_target, th, NUM_RANDOMIZATIONS=100, plot_stats=True, verbose=VERBOSE, DO_RANDOMIZATION_STEP=False)

    #cProfile.run('run()', filename="my.profile")