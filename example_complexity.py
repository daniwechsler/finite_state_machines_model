"""
In this example a random population of N FSM with n states each
is created and the minimization algorithm is used to estimate the
average complexity of the FSM.
"""
from core.finite_state_machine import *
from core.utilities import *
from core.basic_measures import *
from core.modularity import *
from core.fsm_minimization import *

N = 20  # Number of nodes
n = 10  # Number of FSM states
th = 1.0

# Create a population of N randomly sampled FSM
population = get_random_population(N, n)

# Compute the adjacency matrix G
G = compute_G(population, th=th, return_cycles=False, return_pins=False)

printMatrix(G)

#################################################
# COMPLEXITY
#################################################
print()
print("Avg. complexity (before mimimization): ", np.mean([fsm.n for fsm in population]))

avg_complexity, minimizer = minimize_fsm(population, th=1.0, MAX_NO_IMPROVEMENT=100, return_minimizer=True)
print("Avg. complexity (after mimimization): ", avg_complexity)
print(minimizer.fsm_complexities)

# Calculate network after minimzation
G_min = compute_G(minimizer.fsm_, th=1.0, return_cycles=False, return_pins=False)
printMatrix(G_min)