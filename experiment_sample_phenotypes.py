"""
Samples genotypes and evaluates their phenotypes. The script creates a table where each row corresponds to a phenotype
with the column 'count' indicating the number of genotypes with that phenotype (in the sample) and another table that
records the current number of samples whenever a new phenotype is discovered.
"""

import sys
import pandas as pd
import numpy as np
import os
from core.finite_state_machine import get_random_population
from core.utilities import compute_G

OUT_DIR = 'data/genotype_phenotype/th05/'

MAX_NUM_SAMPLES = 150000
MIN_NUM_SAMPLES = 0
N_ = [20]
n_ = [3, 9, 27]
th = 0.5

def get_phenotype_sample(N, n, th):

    max_num_phenotypes = 2**N
    df_phenotypes = pd.DataFrame(columns=['phenotype', 'count'])
    df_discovery_times = pd.DataFrame(columns=['sample_number', 'num_phenotypes'])
    population = get_random_population(N, n, alphabet=['0', '1'])

    for i in range(MAX_NUM_SAMPLES):

        if i % 1000 == 0: print(f"{round(i/MAX_NUM_SAMPLES*100,1)}%")
        if len(df_phenotypes) == max_num_phenotypes and i > MIN_NUM_SAMPLES:
            break

        individual = get_random_population(1, n, alphabet=['0', '1'])[0]
        A = compute_G(population + [individual], th=th)
        phenotype = "".join([str(s) for s in list(A[N,0:N])])
        found = df_phenotypes[df_phenotypes['phenotype'] == phenotype]

        if len(found) == 1:
            df_phenotypes.at[found.index[0], 'count'] += 1
        else:
            df_discovery_times.loc[len(df_discovery_times), :] = [i+1,  len(df_phenotypes)]
            df_phenotypes.loc[len(df_phenotypes), :] = [phenotype, 1]

    return df_phenotypes, df_discovery_times


for n in n_:
    for N in N_:

        dir = OUT_DIR + f"N{N}_n{n}_{MAX_NUM_SAMPLES}/"
        if os.path.isdir(dir):
            print(f"Directory {dir} already exists.")
            continue
        else:
            os.mkdir(dir)

            df_phenotypes, df_discovery_times = get_phenotype_sample(N, n, th)
            df_phenotypes.to_csv(path_or_buf=dir + "phenotypes.csv")
            df_discovery_times.to_csv(path_or_buf=dir + "discovery_times.csv")


