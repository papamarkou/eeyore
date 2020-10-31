# %% Power posterior sampling of MLP weights using XOR data
#
# Sampling the weights of a multi-layer perceptron (MLP) using the XOR data and power posterior algorithm.

# %% Import packages for MCMC simulation

import torch

from torch.distributions import Normal
from torch.utils.data import DataLoader

from eeyore.constants import loss_functions
from eeyore.datasets import XYDataset
from eeyore.models import mlp
from eeyore.samplers import PowerPosteriorSampler

# %% Load XOR data

xor = XYDataset.from_eeyore('xor', dtype=torch.float32)
dataloader = DataLoader(xor, batch_size=len(xor))

# %% Setup MLP model

hparams = mlp.Hyperparameters(dims=[2, 2, 1])

model = mlp.MLP(loss=loss_functions['binary_classification'], hparams=hparams, dtype=torch.float32)

model.prior = Normal(
    torch.zeros(model.num_params(), dtype=model.dtype),
    (3 * torch.ones(model.num_params(), dtype=model.dtype)).sqrt()
)

# %% Setup PowerPosteriorSampler

num_chains = 10

per_chain_samplers = [['MALA', {'step': 0.02}] for _ in range(num_chains)]

sampler = PowerPosteriorSampler(
    model,
    dataloader,
    per_chain_samplers,
    theta0=model.prior.sample(),
    temperature=[1. for _ in range(num_chains)],
    between_step=1,
    check_input=True
)

# %% Run PowerPosteriorSampler

sampler.run(num_epochs=1100, num_burnin_epochs=100, verbose=True, verbose_step=100)

# %% Import kanga package for visual MCMC summaries

import kanga.plots as ps

# %% Generate kanga ChainArray from eeyore ChainList

chain_array = sampler.get_chain().to_kanga()

# %% Compute Monte Carlo mean

print('Monte Carlo mean: {}'.format(chain_array.mean()))

# %% Compute Monte Carlo standard error

print('Monte Carlo standard error: {}'.format(chain_array.mc_se()))

# %% Compute multivariate ESS

print('Multivariate ESS: {}'.format(chain_array.multi_ess()))

# %% Plot traces of simulated Markov chain

for i in range(model.num_params()):
    ps.trace(
        chain_array.get_param(i),
        title=r'Traceplot of $\theta_{{{}}}$'.format(i+1),
        xlabel='Iteration',
        ylabel='Parameter value'
    )

# %% Plot running means of simulated Markov chain

for i in range(model.num_params()):
    ps.running_mean(
        chain_array.get_param(i),
        title=r'Running mean plot of parameter $\theta_{{{}}}$'.format(i+1),
        xlabel='Iteration',
        ylabel='Running mean'
    )

# %% Plot histograms of marginals of simulated Markov chain

for i in range(model.num_params()):    
    ps.hist(
        chain_array.get_param(i),
        bins=30,
        density=True,
        title=r'Histogram of parameter $\theta_{{{}}}$'.format(i+1),
        xlabel='Parameter value',
        ylabel='Parameter relative frequency'
    )
