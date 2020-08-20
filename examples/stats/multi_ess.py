# Compute multivariate ESS using multi_ess function based on eeyore

# %% Load packages

import numpy as np
import torch

from eeyore.stats import multi_ess

# %% Read chains

chains = torch.as_tensor(np.genfromtxt('chain01.csv', delimiter=','))

# %% Compute multivariate ESS using INSE MC covariance estimation

ess_val = multi_ess(chains)

print('Multivariate ESS using INSE MC covariance estimation: {}'.format(ess_val))
