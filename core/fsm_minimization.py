"""
Script to minimize a set of FSM such that their interactions are preserved.
"""

import numpy as np


class MinimizerUnipartite:
    """
      Given a set of FSM, the Minimizer tries to find a new set of FSM that:

      (1) Gives rise to the same adjacency matrix G as the original set.
      (2) Has the minimum number of FSM states (i.e. the sum of the number of
      states of the FSM in the set is minimal).

      It does so by iteratively replacing an existing FSM i with a mutated version
      (rewiring, re-labeled, deleted states) i' that behave identically as i
      (with respect to encounters with FSM from the other group/set).

      """

    STATE_DELETION_PROBABILITY = 0.5

    def __init__(self, fsm_, mu=0.01, th=1.0):
        """
        :param fsm_:
        :param mu:
        """

        # Copy to make sure that internal modifications do not affect external stuff
        self.fsm_ = [fsm_i.clone() for fsm_i in fsm_]

        self.G = np.zeros((len(fsm_), len(fsm_)), dtype=int)
        self.th = th
        self.compute_G(fsm_)

        self.mu = mu
        self.fsm_complexities = [fsm_i.n for fsm_i in fsm_]
        self.E = np.sum(self.fsm_complexities)
        self.trace = dict()
        self.minimize_individual_i = None

    def compute_G(self, fsm_):
        for i, fsm_i in enumerate(fsm_):
            for j, fsm_j in enumerate(fsm_):
                self.G[i, j], out1, out2, pin1, pin2 = fsm_i.encounter(fsm_j, th=self.th, return_cycle=True, return_pin=True)

    def select_fsm_for_change(self):
        """
        Returns the index of a randomly chosen fsm.
        :return:
        """

        # In case only a specific individual should be minimized. See: minimize_individual(..)
        if not self.minimize_individual_i is None:
            return self.minimize_individual_i

        i = np.random.randint(0, self.G.shape[0])
        return i

    def same_G(self, i, fsm_i):
        """
        Returns true if a replacement of the FSM with index i by fsm_i does
        not lead to a change in the interaction matrix. Otherwise false is returned.
        :param i: The index of the FSM that was changed
        :param fsm_i: The changed version of FMS i
        :return:
        """
        for j in range(self.G.shape[0]):
            fsm_j = self.fsm_[j]
            if i == j:
                G_ij_new = fsm_i.encounter(fsm_i, th=self.th)
            else:
                G_ij_new = fsm_i.encounter(fsm_j, th=self.th)
            if G_ij_new != self.G[i, j]:
                return False
        return True

    def get_perturbed_fsm(self, fsm):
        """
        Returns a copy of fsm with some random modifications.
        :return:
        """
        fsm_tmp = fsm.clone()

        if np.random.random() < (1.0 - self.STATE_DELETION_PROBABILITY):
            fsm_tmp.mutate(self.mu)
        else:
            if fsm_tmp.n > 1:
                fsm_tmp = fsm_tmp.delete_state(np.random.randint(0, fsm.n))
        return fsm_tmp

    def update(self):

        # Select random FSM
        i = self.select_fsm_for_change()
        fsm_i = self.fsm_[i]

        fsm_i_tmp = self.get_perturbed_fsm(fsm_i)

        fsm_complexities_tmp = self.fsm_complexities[:]
        fsm_complexities_tmp[i] = fsm_i_tmp.n
        E_new = np.sum(fsm_complexities_tmp)

        if self.same_G(i, fsm_i_tmp) and E_new <= self.E:

            self.fsm_[i] = fsm_i_tmp
            self.fsm_complexities = fsm_complexities_tmp

            if self.E > E_new:
                self.E = E_new
                return True
        return False

    def minimize(self, MAX_NO_IMPROVEMENT=10000, MAX_UPDATES=None, verbose=False):

        num_no_improvement = 0
        steps = 0
        while num_no_improvement < MAX_NO_IMPROVEMENT:

            if self.update():
                num_no_improvement = 0
                if verbose:
                    print("Energy: ", self.E)
                self.trace[steps] = float(self.E)
            else:
                num_no_improvement += 1

            if not MAX_UPDATES is None and steps >= MAX_UPDATES:
                self.trace[steps] = self.E
                break

            steps += 1

    def minimize_individual(self, i, MAX_NO_IMPROVEMENT=10000, MAX_UPDATES=None, verbose=False):
        """
        Minimize only a specific FSM and leave all other FSM unchanged.
        :param fsm_i:
        :param axis:
        :param MAX_NO_IMPROVEMENT:
        :param MAX_UPDATES:
        :param verbose:
        :return:
        """
        self.minimize_individual_i = i
        self.minimize(MAX_NO_IMPROVEMENT=MAX_NO_IMPROVEMENT, MAX_UPDATES=MAX_UPDATES, verbose=verbose)


def minimize_fsm(fsm_, th=1.0, MAX_NO_IMPROVEMENT=10000, verbose=False, return_minimizer=False):

    minim = MinimizerUnipartite(fsm_, th=th)
    minim.minimize(MAX_NO_IMPROVEMENT=MAX_NO_IMPROVEMENT, verbose=verbose)
    avg_rel_complexity = minim.E / len(fsm_)
    if return_minimizer:
        return avg_rel_complexity, minim
    return avg_rel_complexity,