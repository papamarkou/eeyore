# Compute multivariate potential scale reduction factor (Rhat) using multi_rhat function of eeyore

# %% Load packages

import numpy as np
import torch

from eeyore.stats import multi_rhat

# %% Read chains

chains = torch.as_tensor(np.array([np.genfromtxt('chain'+str(i+1).zfill(2)+'.csv', delimiter=',') for i in range(4)]))

# %% Compute multivariate Rhat

rhat_val, _, _ = multi_rhat(chains)

print('Multivariate Rhat: {}'.format(rhat_val))
