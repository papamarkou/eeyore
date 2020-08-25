# Examples of empirical covariance matrix computation using cov function of eeyore

# %% Load packages

import numpy as np
import torch

from eeyore.stats import cov

# %% Read chains

chains = torch.as_tensor(np.genfromtxt('chain01.csv', delimiter=','))

num_iters, num_pars = chains.shape

# %% Compute covariance matrix

np_cov_matrix = cov(chains)

print('Covariance matrix based on eeyore cov function:\n{}'.format(np_cov_matrix))
