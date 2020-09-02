# %% MALA sampling of MLP weights using XOR data
#
# Learn the XOR function by sampling the weights of an MLP via MALA and store chain in file.

# %% Import packages for MCMC simulation

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
    (3 * torch.ones(model.num_params(), dtype=model.dtype)).sqrt()
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

chain_list = sampler.chain.to_chainlist()

# %% Import kanga package for visual MCMC summaries

import kanga.plots as ps

# %% Generate kanga ChainArray from eeyore ChainList

chain_array = chain_list.to_kanga()

# %% Compute acceptance rate

print('Acceptance rate: {}'.format(chain_array.acceptance_rate()))

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
