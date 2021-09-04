"""
This example creates a random population of N FSM with n states
each and computes the modularity of the resulting interaction
network.
"""
from core.finite_state_machine import *
from core.utilities import *
from core.basic_measures import *
from core.modularity_maximization import *
from core.modularity import *

N = 20  # Number of nodes
n = 8  # Number of FSM states
th =1.0

# Create a population of N randomly sampled FSM
population = get_random_population(N, n)

# Compute the adjacency matrix G
G = compute_G(population, th=th, return_cycles=False, return_pins=False)

printMatrix(sort_matrix_by_modules(G, modularity_function=cacl_modularity_unipartite))

#print_cylce_and_pin_matrix(population)

# Calc some basic stats of the network
connectance = calc_connectance(G)
degrees = calc_degrees(G)
Q = modularity(G, return_groups=False)

print("degees: ", degrees)
print("c: ", connectance)
print("Q: ", Q)

# Compute max modularity (preserving node degrees)
Q_max_1, G_max_1 =  calc_max_modularity(G, MAX_NO_IMPROVEMENT=200, swapping_function=swap_unipartite)

printMatrix(sort_matrix_by_modules(G_max_1, modularity_function=cacl_modularity_unipartite))

print("degees: ", calc_degrees(G_max_1))
print("c: ", calc_connectance(G_max_1))
print("Q_max: ", Q_max_1)

# Compute max modularity (preserving connectance)
Q_max_2, G_max_2 =  calc_max_modularity(G, MAX_NO_IMPROVEMENT=200, swapping_function=rewire_unipartite)
printMatrix(sort_matrix_by_modules(G_max_2, cacl_modularity_unipartite))
print("degees: ", calc_degrees(G_max_2))
print("c: ", calc_connectance(G_max_2))
print("Q_max: ", Q_max_2)

# Compute normalized modularity (can take a while)
Q_stats = cacl_modularity_unipartite(G, NUM_RANDOMIZATIONS=40, RETURN_NORMALIZED=True, MODULARITY_NORM_MAX_NO_IMPROVEMENT=200)
print("Q_norm: ", Q_stats['Q_norm'])

