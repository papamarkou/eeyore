# %% MALA sampling of MLP weights using XOR data
#
# Learn the XOR function by sampling the weights of an MLP via MALA and store chain in file.

# %% Import packages

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch

from pathlib import Path
from torch.distributions import Normal
from torch.utils.data import DataLoader

from eeyore.chains import ChainFile
from eeyore.constants import loss_functions
from eeyore.datasets import XYDataset
from eeyore.models import mlp
from eeyore.samplers import MALA

# %% Load XOR data

xor = XYDataset.from_eeyore('xor')
dataloader = DataLoader(xor, batch_size=len(xor))

# %% Setup MLP model

hparams = mlp.Hyperparameters(dims=[2, 2, 1])
model = mlp.MLP(loss=loss_functions['binary_classification'], hparams=hparams)
model.prior = Normal(
    torch.zeros(model.num_params(), dtype=model.dtype),
    np.sqrt(3)*torch.ones(model.num_params(), dtype=model.dtype)
)

# %% Setup chain

chain = ChainFile(
    keys=['sample', 'target_val', 'grad_val', 'accepted'],
    path=Path.cwd().joinpath('output')
)

# %% Setup MALA sampler

sampler = MALA(model, theta0=model.prior.sample(), dataloader=dataloader, step=1.74, chain=chain)

# %% Run MALA sampler

sampler.run(num_epochs=11000, num_burnin_epochs=1000)

# %% Convert ChainFile instance to ChainList instance

chainlist = sampler.chain.to_chainlist()

# %% Compute acceptance rate

print('Acceptance rate: {}'.format(chainlist.acceptance_rate()))

# %% Compute Monte Carlo mean

print('Monte Carlo mean: {}'.format(chainlist.mean()))

# %% Plot traces of simulated Markov chain

for i in range(model.num_params()):
    chain = chainlist.get_sample(i)
    plt.figure()
    sns.lineplot(range(len(chain)), chain)
    plt.xlabel('Iteration')
    plt.ylabel('Parameter value')
    plt.title(r'Traceplot of $\theta_{{{0}}}$'.format(i+1))

# %% Plot running means of simulated Markov chain

for i in range(model.num_params()):
    chain = chainlist.get_sample(i)
    chain_mean = torch.empty(len(chain))
    chain_mean[0] = chain[0]
    for j in range(1, len(chain)):
        chain_mean[j] = (chain[j]+j*chain_mean[j-1])/(j+1)

    plt.figure()
    sns.lineplot(range(len(chain)), chain_mean)
    plt.xlabel('Iteration')
    plt.ylabel('Parameter value')
    plt.title(r'Running mean of $\theta_{{{0}}}$'.format(i+1))

# %% Plot histograms of simulated Markov chain

for i in range(model.num_params()):
    plt.figure()
    sns.distplot(chainlist.get_sample(i), bins=20, norm_hist=True)
    plt.xlabel('Value range')
    plt.ylabel('Relative frequency')
    plt.title(r'Histogram of $\theta_{{{0}}}$'.format(i+1))
