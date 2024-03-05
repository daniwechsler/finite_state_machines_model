"""
Runs experiment w defined in the manuscript.

It evolves a community of N FSM (having n states each) until the interaction network has a desired modularity (Q_target)
and a desired connectance (c_target).

The execution of the script is time-consuming. Runtime depends especially on N, n and the choice of the parameters for
the modularity calculation, complexity calculation and community evolution algorithms.
"""

from core.network_evolution import run_modularity_optimization

n = 10
N = 20
th = 1.0

c_target = 0.3
Q_target = 0.3

# Settings for modularity algorithm
# Value used in experiments: 500
NUM_RANDOMIZATIONS = 50
# Value used in experiments: 20000
MAX_MODULARITY_MAX_NO_IMPROVEMENT = 1000

# Settings for complexity algorithm
# Value used in experiments: 2000000
FSM_COMPLEXITY_MAX_NO_IMPROVE = 20000

# Settings for the network evolution algorithm
MAX_UPDATES = 200000
C_EPSILON = 0.0025
Q_NORM_EPSILON = 0.005
MUT_EPSILON = 0.0001
NUM_Q_MAX_SAMPLES = 5

# Evolve the community to the target modularity and target connectance
stats = run_modularity_optimization(n, N, c_target, Q_target, th,
                                    MAX_UPDATES=MAX_UPDATES,
                                    FSM_COMPLEXITY_MAX_NO_IMPROVE=FSM_COMPLEXITY_MAX_NO_IMPROVE,
                                    NUM_RANDOMIZATIONS=NUM_RANDOMIZATIONS,
                                    MAX_MODULARITY_MAX_NO_IMPROVEMENT=MAX_MODULARITY_MAX_NO_IMPROVEMENT,
                                    C_EPSILON=C_EPSILON,
                                    Q_NORM_EPSILON=Q_NORM_EPSILON,
                                    MUT_EPSILON=MUT_EPSILON,
                                    NUM_Q_MAX_SAMPLES=NUM_Q_MAX_SAMPLES,
                                    )

print(stats)
