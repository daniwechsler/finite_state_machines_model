"""
In this example a population of N FSM with n states each is
evolved such that the interaction network has a desired connectance
and modularity.
"""
from core.network_evolution import *

N = 20  # Number of nodes
n = 10  # Number of FSM states
c_target = 0.3 # Desired connectance
Q_norm_target = 0.1 # Desired normalized modularity
th = 1.0 # Interaction specificity

run_modularity_optimization(n, N, c_target, Q_norm_target, th,
                                MAX_UPDATES = 8000,
                                NUM_RANDOMIZATIONS=20,
                                MAX_MODULARITY_MAX_NO_IMPROVEMENT=200,
                                FSM_COMPLEXITY_MAX_NO_IMPROVE=500,
                                DO_RANDOMIZATION_STEP=True,
                                C_EPSILON=0.0025,
                                Q_NORM_EPSILON = 0.001,
                                MUT_EPSILON = 0.001,
                                NUM_Q_MAX_SAMPLES = 1,
                                plot_stats=False,
                                verbose=True)
