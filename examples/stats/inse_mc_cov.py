# Compute INSE Monte Carlo covariance estimate using inse_mc_cov function on eeyore

# %% Load packages

import numpy as np
import torch

from kanga.stats import inse_mc_cov

# %% Read chains

chains = torch.as_tensor(np.genfromtxt('chain01.csv', delimiter=','))

# %% Compute INSE Monte Carlo covariance estimate

inse_mc_cov_val = inse_mc_cov(chains)

print('INSE Monte Carlo covariance estimate:\n{}'.format(inse_mc_cov_val))

# %% Compute adjusted INSE Monte Carlo covariance estimate

adj_inse_mc_cov_val = inse_mc_cov(chains, adjust=True)

print('Adjusted INSE Monte Carlo covariance estimate:\n{}'.format(adj_inse_mc_cov_val))
