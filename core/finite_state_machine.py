"""
Implementation of the finite-state machine (FSM) class.
"""

import numpy as np
import copy
import itertools as it


class FSM(object):
    """
       0    1   2     3    4   5     6    7   8
    +===========================================+
    | '1' | 2 | 2 || '0' | 2 | 1 || '1' | 0 | 0 |
    +===========================================+

    To improve performance, this genome representation is converted to
    a transition table of the following structure (implemented as a dictionary).

     S_now    in_symbol  |  S_next   out_symbol
    -------------------------------------------
       0      '0'        |    2      '1'
       0      '1'        |    2      '1'
       1      '0'        |    2      '1'
       1      '1'        |    1      '0'
       2      '0'        |    0      '1'
       2      '1'        |    0      '1'

    S_now       := The current state
    in_symbol   := Input symbol
    S_next      := The state to transfer to when in symbol is received in S_now
    out_symbol  := The symbol of S_now
    """

    def __init__(self, genome, n, symbols):
        """

        :param genome:
        :param n:
        :param symbols: A dictionary with each possible symbol being
        one key and the corresponding value being the position on the
        genome where the state to transfer to is indicated when the
        symbol is received. Example for three symbols a,b,c:

        symbols = {'a': 0, 'b': 1, 'c': 2}
        """
        self.n = n
        self.symbols = symbols

        self.genome = genome
        self.alphabet_size = len(symbols)
        self.state = 0
        self.current_symbol = self.genome[self.state * (self.alphabet_size + 1)]
        self.transition_table = None
        # Stores a minimized version of this FSM that can be used to calculate encounters faster
        self.minimized_fsm = None
        assert len(genome) == n + n * self.alphabet_size

        self.create_transition_table()

    def create_transition_table(self):

        self.transition_table = dict()

        for state in range(self.n):
            for symbol in self.symbols:
                target_state = self.genome[state * (self.alphabet_size + 1) + self.symbols[symbol] + 1]
                out_symbol = self.genome[target_state * (self.alphabet_size + 1)]
                self.transition_table[(state, symbol)] = (target_state, out_symbol)

    def reset(self):
        self.state = 0
        self.current_symbol = self.genome[self.state * (self.alphabet_size + 1)]

    def communicate(self, other, return_pin=False):
        """
        Let's self communicate with other. Both FSM start in the initial state and
        communication proceeds until a cycle is reached.
        The method returns for both FSM a list with the symbols sent to the other
        FSM while on the cycle.
        If return_pin is true, for each FSM also a list with the symbols it sent before
        reaching the cycle is returned.

        :param other:
        :param return_pin:
        :return:
        """
        states = []
        out_self = []
        out_other = []
        self.reset()
        other.reset()

        while True:
            state_pair = (self.state, other.state)

            if state_pair in states:
                # Find the iteration at which the state pair occurred the first time.
                index = states.index(state_pair)

                if not return_pin:
                    return out_self[index:], out_other[index:]

                return out_self[index:], out_other[index:], out_self[0:index], out_other[0:index:]

            states.append(state_pair)

            out_self.append(self.current_symbol)
            out_other.append(other.current_symbol)

            self_current_symbol_tmp = self.current_symbol
            other_current_symbol_tmp = other.current_symbol

            self.iterate(other_current_symbol_tmp)
            other.iterate(self_current_symbol_tmp)

    def encounter(self, other, th=1.0, return_cycle=False, return_pin=False, minimize=False):
        # Use a minimized version of the two FSM to compute if they interact.
        # It is an experimental feature. Test showed little performance improvement.
        if minimize:
            out1, out2, pin1, pin2 = self.get_minimized().communicate(other.get_minimized(), return_pin=True)
        else:
            out1, out2, pin1, pin2 = self.communicate(other, return_pin=True)

        num_comm = sum(int(out1[o] == out2[o]) for o in range(len(out1)))
        comm_symb = num_comm / len(out1)
        interact = int(comm_symb >= th)

        if return_cycle == return_pin == False:
            return interact

        ret = (interact,)

        if return_cycle:
            ret = ret + (out1, out2)

        if return_pin:
            ret = ret + (pin1, pin2)

        return ret

    def iterate(self, input_symbol):
        self.state, self.current_symbol = self.transition_table[(self.state, input_symbol)]

    def get_next_state(self, state, input_symbol):
        next_state, next_symbol = self.transition_table[(state, input_symbol)]
        return next_state

    def get_current_symbol(self):
        return self.current_symbol

    def get_symbol_of_state(self, state):
        return self.genome[state * (self.alphabet_size + 1)]

    def mutate(self, mu):
        """
        Swaps each entry in the genome with probability mu to one of the allowed
        values (the new value is chosen uniformly at random from the allowed values)
        :param mu:
        :return:
        """
        for i, g in enumerate(self.genome):
            if np.random.random() < mu:
                if i % (self.alphabet_size+1) == 0:
                    # Mutate state symbol
                    self.genome[i] = np.random.choice(list(self.symbols.keys()))
                else:
                    # Mutate transition
                    self.genome[i] = np.random.randint(0, self.n)

        self.create_transition_table()

    def mutate_gene(self, i):
        """
        Changes gene i to a randomly chosen value that is not equal
        to the current value.
        :param i:
        :return:
        """

        if i % (self.alphabet_size + 1) == 0:
            # Mutate state symbol
            allowed_values = list(self.symbols.keys())
        else:
            # Mutate transition
            allowed_values = list(range(0, self.n))

        allowed_values.remove(self.genome[i])
        self.genome[i] = np.random.choice(allowed_values)

        self.create_transition_table()

    def duplicate_state(self, state):
        """
        Creates a new FSM from this one by duplicating the given state. The duplicated
        state will have the same output edges and each input edge to the given state
        will be rewired to the new state with p=0.5.
        :return:
        """
        assert state < self.n
        new_genome = copy.deepcopy(self.genome)
        successor_states = self.get_successor_states(state)

        # Make sure that if state has self connections these connections are rewired for the duplicated node
        # to point to the duplicate.
        for i in range(self.alphabet_size):
            if successor_states[i] == state:
                successor_states[i] = self.n

        new_genome += [self.genome[self.state * (self.alphabet_size + 1)]] + successor_states
        for s in range(self.n):
            if s == state:
                continue
            for i in range(s*(self.alphabet_size + 1)+1, s*(self.alphabet_size + 1)+1 + self.alphabet_size):
                if new_genome[i] == state:
                    new_genome[i] = np.random.choice([state, self.n])

        return FSM(new_genome, self.n+1, copy.deepcopy(self.symbols))

    def delete_state(self, state):
        """
        Creates a new FSM from this one with the given state deleted. The ingoing edges of the deleted
        node are rewired do random nodes.
        :param state:
        :return:
        """
        assert state < self.n

        # The state ids of all states with id larger then the id of the
        # state that is deleted must be decremented by 1.
        node_id_map = list(range(0, state)) + [None] + list(range(state, self.n-1))
        new_genome = copy.deepcopy(self.genome)

        for s in range(self.n):
            for i in range(s*(self.alphabet_size + 1)+1, s*(self.alphabet_size + 1)+1 + self.alphabet_size):
                if node_id_map[new_genome[i]] is None:
                    new_genome[i] = np.random.randint(0, self.n-1)
                else:
                    new_genome[i] = node_id_map[new_genome[i]]

        new_genome = new_genome[0: state*(self.alphabet_size + 1)] + new_genome[state*(self.alphabet_size + 1)+
                                                                                (self.alphabet_size + 1):]

        return FSM(new_genome, self.n - 1, copy.deepcopy(self.symbols))

    def clone(self):

        return FSM(self.genome[:], self.n, copy.deepcopy(self.symbols))

    def get_successor_states(self, state):
        """
        Returns a list of the succeeding states of the given state.
        The list has the same order as on the genome.
        :param state:
        :return:
        """
        return self.genome[state * (self.alphabet_size + 1) + 1: state * (
                    self.alphabet_size + 1) + 1 + self.alphabet_size]

    def find_reachable_states(self, state=0, visited_states=None):
        """
        Finds all states reachable from the given starting state. Returns a list containing
        all reachable states.
        :param state:
        :param visited_states:
        :return:
        """
        if visited_states is None:
            visited_states = []

        visited_states.append(state)

        # Find set of states directly succeeding the given state
        successor_states = self.get_successor_states(state)

        # For each of the succeeding states not already visited find the states reachable by them
        for s_state in successor_states:
            if not s_state in visited_states:
                visited_states = self.find_reachable_states(s_state, visited_states)

        return visited_states

    def remove_unreachable_states(self):
        """
        Returns an FSM that behaves as self but with all non reachable states removed.
        :return:
        """
        reachable_states = sorted(self.find_reachable_states())

        new_genome = []
        for state in reachable_states:

            new_genome.append(self.genome[state*(self.alphabet_size + 1)])
            successor_states = self.get_successor_states(state)
            for successor_state in successor_states:
                new_genome.append(reachable_states.index(successor_state))

        return FSM(new_genome, len(reachable_states), copy.deepcopy(self.symbols))

    def get_minimized(self):
        if self.minimized_fsm is None:
            self.minimized_fsm = self.minimize()
        return self.minimized_fsm

    def minimize(self):
        """
        Returns an FSM that behaves as self but uses the minimum possible number of states.

        Implementation of the pseudo code in:
        en.wikipedia.org/wiki/DFA_minimization#CITEREFHopcroftMotwaniUllman2001
        :return:
        """

        # Check if there are unreachable states. If so, first create an FSM with
        # the unreachable states removed and call minimize() on this new FSM.
        # Otherwise proceed in this (self) FSM
        if len(self.find_reachable_states()) < self.n:
            fsm = self.remove_unreachable_states()
            return fsm.minimize()

        P = []
        W = []

        # Spilt states into two sets

        state_partitions_0 = dict()
        for symbol in self.symbols:
            state_partitions_0[symbol] = []

        for state in range(self.n):
            symbol = self.genome[state * (self.alphabet_size + 1)]
            state_partitions_0[symbol].append(state)

        for symbol in self.symbols:
            if len(state_partitions_0[symbol]) > 0:
                P.append(state_partitions_0[symbol])
                W.append(state_partitions_0[symbol])

        while len(W) > 0:

            A = W.pop(0)
            for symbol in self.symbols:
                P_updated = []
                for Y in P:
                    # Group states in Y according to whether they transition to a state in A
                    # upon receiving symbol or not.
                    transition_to_A = []
                    transition_not_to_A = []
                    for state in Y:
                        if self.transition_table[(state, symbol)][0] in A:
                            transition_to_A.append(state)
                        else:
                            transition_not_to_A.append(state)

                    if len(transition_to_A) == 0 or len(transition_not_to_A) == 0:
                        P_updated.append(Y)
                        continue
                    else:
                        P_updated.append(transition_to_A)
                        P_updated.append(transition_not_to_A)
                        if Y in W:
                            W.remove(Y)
                            W.append(transition_to_A)
                            W.append(transition_not_to_A)
                        else:
                            if len(transition_to_A) < len(transition_not_to_A):
                                W.append(transition_to_A)
                            else:
                                W.append(transition_not_to_A)
                P = P_updated

        # Create a mapping from the old state ids to the new state is.
        # First, make sure that set of states that contains state 0 is at first position in P.
        for i, Y in enumerate(P):
            if 0 in Y:
                break

        tmp = P[0]
        P[0] = P[i]
        P[i] = tmp

        state_map = [None] * self.n
        for i, Y in enumerate(P):
            for old_state in Y:
                state_map[old_state] = i

        new_genome = []
        for i, Y in enumerate(P):
            old_state = Y[0]
            new_genome.append(self.genome[old_state * (self.alphabet_size + 1)])
            successor_states = self.get_successor_states(old_state)
            for successor_state in successor_states:
                new_genome.append(state_map[successor_state])

        fsm_minimized = FSM(new_genome, len(P), copy.deepcopy(self.symbols))

        return fsm_minimized

    def get_neighbouring_FSM(self):
        """
        Returns a list of all FSM that differ in a single mutation from self.
        :return:
        """
        neighbours = []
        for i in range(len(self.genome)):

            if i % (self.alphabet_size + 1) == 0:
                for symbol in self.symbols:
                    if symbol == self.genome[i]: continue
                    genome_p = self.genome[:]
                    genome_p[i] = symbol
                    neighbour = FSM(genome_p, self.n, copy.deepcopy(self.symbols))
                    neighbours.append(neighbour)
            else:
                for state in range(self.n):
                    if state == self.genome[i]: continue
                    genome_p = self.genome[:]
                    genome_p[i] = state
                    neighbour = FSM(genome_p, self.n, copy.deepcopy(self.symbols))
                    neighbours.append(neighbour)

        return neighbours

    def to_adjacency_list(self):
        """
        Returns the graph underlying the FSM as an adjacency list.
        Which is represented as a dictionary with keys being nodes and
        values lists of nodes to which a node has an outgoing edge.
        :return:
        """
        adjacency_list = {}
        for s in range(self.n):
            adjacency_list[s] = []
            for i in range(s * (self.alphabet_size + 1) + 1, s * (self.alphabet_size + 1) + 1 + self.alphabet_size):
                adjacency_list[s].append(self.genome[i])
        return adjacency_list

    def __eq__(self, other):

        if self.n != other.n:
            return False

        for i in range(len(self.genome)):
            if self.genome[i] != other.genome[i]:
                return False
        return True

    def print_transition_table(self):

        print()
        header = "S_t \t" + "\t".join([symbol for symbol in self.symbols]) + "\t Symbol"
        print(header)
        print("-" * len(header.expandtabs()))
        for state in range(self.n):
            successor_states = self.get_successor_states(state)
            row = str(state) + "\t" + "\t".join([str(s_next) for s_next in successor_states]) + "\t " + self.genome[state * (self.alphabet_size + 1)]
            print(row)


def get_random_genome(n, alphabet=["0", "1"], accept_state=False, random_generator=None):
    """
    Returns a genome (sequential) representation of a finite-state machine.
    Each of the n states is represented by 1 + len(alphabet) digits (i.e., 'nucleotides').
    The first one indicates the output symbol of the state and the remaining len(alphabet)
    digits indicate for each possible symbol the next state to transition to when receiving
    that symbol. The starting state is the left most state on the genome.

    A genome for a n=3 and alphabet=[0, 1] may look like this:

        S1      S2      S3
     +------+-------+------+
        v v     v v     v v
      1 0 2 - 0 2 1 - 1 1 2
      ^       ^       ^

    ^ := output symbols
    v := target states

    :param n:
    :param alphabet:
    :param accept_state:
    :param random_generator:
    :return:
    """
    if not random_generator is None:
        rg = random_generator
    else:
        rg = np.random
    genome = []
    symbols = dict()
    for j in range(len(alphabet)):
        symbols[alphabet[j]] = j

    for i in range(n):
        genome.append(alphabet[rg.randint(len(alphabet))])
        for j in range(len(alphabet)):
            genome.append(rg.randint(n))
        if accept_state:
            genome.append(rg.randint(3))

    return genome, symbols


def get_random_population(N, n, alphabet=['0', '1'], mu=None, random_generator=None):
    population = []
    # If mutation rate is given create a seed FSM and N-1 clones
    # with each gene mutated with probability mu.
    # Otherwise, just sample N FSM uniformly at random.

    if not mu is None:
        genome, symbols = get_random_genome(n, alphabet)
        fsm = FSM(genome, n, symbols)
        population.append(fsm)
        for i in range(N-1):
            fsm_clone = fsm.clone()
            fsm_clone.mutate(mu)
            population.append(fsm_clone)
    else:
        for i in range(N):
            genome, symbols = get_random_genome(n, alphabet, random_generator=random_generator)
            fsm = FSM(genome, n, symbols)
            population.append(fsm)
    return population


def enumerate_FSM(n, alphabet=['0', '1']):
    """
    Returns a list with all possible FSM with n states.
    :param n:
    :param alphabet:
    :return:
    """

    FMS_ = []
    state_labelings = list(it.product(alphabet, repeat=n))
    state_transitions = list(it.product(list(range(n)), repeat=n*len(alphabet)))

    for state_labels in  state_labelings:
        for state_transition in state_transitions:
            genome = []
            for i in range(n):
                genome += [state_labels[i]]
                genome += state_transition[i*len(alphabet):i*len(alphabet)+len(alphabet)]

            symbols = dict()
            for j in range(len(alphabet)):
                symbols[alphabet[j]] = j

            FMS_.append(FSM(genome, n, symbols))

    return FMS_


