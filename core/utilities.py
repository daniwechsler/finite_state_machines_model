import numpy as np


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


def print_cylce_and_pin_matrix(population):
    """
    Prints for each pair of FSM in population the symbols they communicate before the cycle
    and on the cycle.
    :param population:
    :return:
    """
    OKGREEN = '\033[34m'
    ENDC = '\033[0m'

    G, out, pins = compute_G(population, return_cycles=True, return_pins=True)

    MAX_PIN_LENGTH = np.max(list(map(len, np.array(pins).flatten())))
    MAX_CYCLE_LENGTH = np.max(list(map(len, np.array(out).flatten())))

    for i in range(len(population)):
        row_animals = ""
        row_plants = ""
        for j in range(len(population)):
            if G[i,j] == 1:
                out_a = pins[i][j].rjust(MAX_PIN_LENGTH) + "|" + OKGREEN + out[i][j].ljust(MAX_CYCLE_LENGTH) + ENDC
                out_p = pins[i][j].rjust(MAX_PIN_LENGTH) + "|" +  OKGREEN + out[i][j].ljust(MAX_CYCLE_LENGTH) + ENDC
            else:
                out_a = pins[i][j].rjust(MAX_PIN_LENGTH) + "|" +  out[i][j].ljust(MAX_CYCLE_LENGTH)
                out_p = pins[i][j].rjust(MAX_PIN_LENGTH) + "|" +  out[i][j].ljust(MAX_CYCLE_LENGTH)

            row_animals += out_a
            row_plants += out_p

        print(row_animals)
        print(row_plants)
        print()

def printMatrix(G, index_start=1, column_gap=2, P_names=None, A_names=None):
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
