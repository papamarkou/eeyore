# %% Metropolis-Hastings sampling of MLP weights using XOR data
# 
# Learn the XOR function by sampling the weights of a multi-layer perceptron (MLP) via the Metropolis-Hastings algorithm.

# %% Import packages

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch

from torch.distributions import Normal
from torch.utils.data import DataLoader

from eeyore.constants import loss_functions
from eeyore.datasets import XYDataset
from eeyore.kernels import NormalKernel
from eeyore.models import mlp
from eeyore.samplers import MetropolisHastings

# %% Load XOR data

xor = XYDataset.from_eeyore('xor', dtype=torch.float64)
dataloader = DataLoader(xor, batch_size=len(xor))

# %% Setup MLP model

hparams = mlp.Hyperparameters(dims=[2, 2, 1])
model = mlp.MLP(loss=loss_functions['binary_classification'], hparams=hparams, dtype=torch.float64)
model.prior = Normal(
    torch.zeros(model.num_params(), dtype=model.dtype),
    np.sqrt(3)*torch.ones(model.num_params(), dtype=model.dtype)
)

# %% Setup Metropolis-Hastings sampler

proposal_scale = 1.55
kernel = NormalKernel(
    torch.zeros(model.num_params(), dtype=model.dtype),
    torch.sqrt(torch.tensor(proposal_scale, dtype=model.dtype)) * torch.ones(model.num_params(), dtype=model.dtype)
)
sampler = MetropolisHastings(model, theta0=model.prior.sample(), dataloader=dataloader, kernel=kernel)

# %% Run Metropolis-Hastings sampler

sampler.run(num_epochs=11000, num_burnin_epochs=1000, verbose=True, verbose_step=1000)

# %% Compute acceptance rate

print('Acceptance rate: {}'.format(sampler.get_chain().acceptance_rate()))

# %% Compute Monte Carlo mean

print('Monte Carlo mean: {}'.format(sampler.get_chain().mean()))

# %% Plot traces of simulated Markov chain

for i in range(model.num_params()):
    chain = sampler.get_sample(i)
    plt.figure()
    sns.lineplot(range(len(chain)), chain)
    plt.xlabel('Iteration')
    plt.ylabel('Parameter value')
    plt.title(r'Traceplot of parameter $\theta_{}$'.format(i+1))

# %% Plot running means of simulated Markov chain

for i in range(model.num_params()):
    chain = sampler.get_sample(i)
    chain_mean = torch.empty(len(chain))
    chain_mean[0] = chain[0]
    for j in range(1, len(chain)):
        chain_mean[j] = (chain[j]+j*chain_mean[j-1])/(j+1)

    plt.figure()
    sns.lineplot(range(len(chain)), chain_mean)
    plt.xlabel('Iteration')
    plt.ylabel('Parameter value')
    plt.title(r'Running mean of parameter {}'.format(i+1))

# %% Plot histograms of marginals of simulated Markov chain

for i in range(model.num_params()):
    plt.figure()
    sns.distplot(sampler.get_sample(i), bins=20, norm_hist=True)
    plt.xlabel('Value range')
    plt.ylabel('Relative frequency')
    plt.title(r'Histogram of parameter {}'.format(i+1))
