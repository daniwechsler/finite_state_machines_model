"""
Runs experiment 1 defined in the manuscript.

For a given combination of network size N and number of FSM states n (and interaction specificity th
(called delta in the manuscript)).

In particular, the script creates a random community containing N FSM (with n states each).
It then infers the interaction network G (by testing for each pair of FSM whether they interact or not).
Finally, it computes modularity (Q) and the average complexity of the FSM.

The script may execute for a while (runtime depends crucially on N, n and the parameter values chosen for the modularity
calculation and fsm minimization algorithms).
"""

from core.finite_state_machine import get_random_population
from core.utilities import compute_G, print_matrix
from core.basic_measures import calc_connectance
from core.modularity import cacl_modularity_unipartite
from core.fsm_minimization import minimize_fsm

n = 10
N = 20
th = 1.0

#############################################################
# Settings for modularity and complexity algorithm
#############################################################

# Value used in experiments: 500
NUM_RANDOMIZATIONS = 20

# Value used in experiments: 20000
MAX_MODULARITY_MAX_NO_IMPROVEMENT = 100

# Value used in experiments: 2000000
FSM_COMPLEXITY_MAX_NO_IMPROVE = 1000

# Sample N FSM uniformly from the set of all FSM with n states
community = get_random_population(N, n)

# Determine the interaction network (adjacency matrix)
G = compute_G(community, th=th)
print_matrix(G)

# Compute connectance
c = calc_connectance(G)
print("Connectance:", c)

# Compute Modularity
Q_stats = cacl_modularity_unipartite(G, NUM_RANDOMIZATIONS=NUM_RANDOMIZATIONS, RETURN_NORMALIZED=True,
                                                MODULARITY_NORM_MAX_NO_IMPROVEMENT=MAX_MODULARITY_MAX_NO_IMPROVEMENT,
                                                RETURN_MODULARITY_MAXIMIZER=True)

Q = Q_stats['Q']    # Raw value of modularity
Q_rand = Q_stats['Q_rand']  # Expected modularity of a random network (same size and connectance as G)
Q_max = Q_stats['Q_max'] # Maximum modularity of a network with same size and connectance as G
Q_norm = Q_stats['Q_norm'] # Normalized modularity (eq. 1)
Q_p_value = Q_stats['p_value'] # P-value of modularity

print("Modularity (norm.):", Q_norm)
print("p-value:", Q_p_value)

# Compute Complexity
avg_complexity, minimizer = minimize_fsm(community, th=1.0, MAX_NO_IMPROVEMENT=FSM_COMPLEXITY_MAX_NO_IMPROVE, return_minimizer=True)
print("Avg. complexity: ", avg_complexity)
# The number of states of each FSM after minimization
print(minimizer.fsm_complexities)

# Calculate network after minimization
G_min = compute_G(minimizer.fsm_, th=1.0, return_cycles=False, return_pins=False)
print_matrix(G_min)

