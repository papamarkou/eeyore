# Examples of empirical correlation matrix computation using cor function of eeyore

# %% Load packages

import numpy as np
import torch

from eeyore.stats import cor

# %% Read chains

chains = torch.as_tensor(np.genfromtxt('chain01.csv', delimiter=','))

num_iters, num_pars = chains.shape

# %% Compute correlation matrix

np_cor_matrix = cor(chains)

print('Correlation matrix based on eeyore cor function:\n{}'.format(np_cor_matrix))
