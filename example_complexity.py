"""
In this example a random population of N FSM with n states each is created and the minimization algorithm is used to
estimate the average complexity of the FSM.
"""
import numpy as np
from core.finite_state_machine import get_random_population
from core.utilities import compute_G, print_matrix
from core.fsm_minimization import minimize_fsm

N = 20  # Number of nodes
n = 10  # Number of FSM states
th = 1.0 # Interaction specificity (delta)
# The maximum number of consecutive iterations without improvement before the algorithm stops
MAX_NO_IMPROVEMENT = 1000

# Create a population of N randomly sampled FSM
population = get_random_population(N, n)

# Compute the adjacency matrix G
G = compute_G(population, th=th, return_cycles=False, return_pins=False)

print_matrix(G)

#################################################
# COMPLEXITY
#################################################
print()
print("Avg. complexity (before mimimization): ", np.mean([fsm.n for fsm in population]))

avg_complexity, minimizer = minimize_fsm(population, th=1.0, MAX_NO_IMPROVEMENT=MAX_NO_IMPROVEMENT, return_minimizer=True)
print("Avg. complexity (after mimimization): ", avg_complexity)
print(minimizer.fsm_complexities)

# Calculate network after minimization
G_min = compute_G(minimizer.fsm_, th=1.0, return_cycles=False, return_pins=False)
print_matrix(G_min)

