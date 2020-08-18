# %% Load packages

import numpy as np
import torch

from eeyore.stats import inse_mc_cov

# %% Read chains

chains_np = np.genfromtxt('chain01.csv', delimiter=',')

x_np = chains_np

x_np = x_np - x_np.mean(0)

# %%

chains = torch.as_tensor(np.genfromtxt('chain01.csv', delimiter=','))

inse_mc_cov(chains)

inse_mc_cov(chains, adjust=True)

# %%

x = chains

x = x - x.mean(0)

# %%

n_np, p_np = x_np.shape

ub_np = int(np.floor(n_np / 2))
sn_np = ub_np

# %%

n, p = x.shape

ub = torch.floor(torch.tensor(n / 2, dtype=x.dtype)).int().item()
sn = ub
