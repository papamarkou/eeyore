# Metropolis-Hastings sampling of logistic regression coefficients using Swiss banknote data
#
# Sampling logistic regression coefficients using the Swiss banknote data and Metropolis-Hastings algorithm

# %% Import packages for MCMC simulation and numerical MCMC summaries and diagnostics

import torch

from datetime import timedelta
from timeit import default_timer as timer
from torch.distributions import Normal
from torch.utils.data import DataLoader

import kanga.plots as ps

from eeyore.datasets import XYDataset
from eeyore.kernels import NormalKernel
from eeyore.models import logistic_regression
from eeyore.samplers import MetropolisHastings
from eeyore.stats import binary_cross_entropy

# %% Load and standardize Swiwss banknote data

banknotes = XYDataset.from_eeyore('banknotes', dtype=torch.float32)
banknotes.x = banknotes.x[: , :4]
banknotes.x = \
    (banknotes.x - torch.mean(banknotes.x, dim=0, keepdim=True))/ \
    torch.std(banknotes.x, dim=0, keepdim=True, unbiased=False)

dataloader = DataLoader(banknotes, batch_size=len(banknotes))

# %% Setup logistic regression model

hparams = logistic_regression.Hyperparameters(input_size=4, bias=False)
model = logistic_regression.LogisticRegression(
    loss=lambda x, y: binary_cross_entropy(x, y, reduction='sum'),
    hparams=hparams,
    dtype=torch.float32
)
model.prior = Normal(
    torch.zeros(model.num_params(), dtype=model.dtype),
    (3 * torch.ones(model.num_params(), dtype=model.dtype)).sqrt()
)

# %% Setup Metropolis-Hastings sampler

proposal_scale = 0.5
kernel = NormalKernel(
    torch.zeros(model.num_params(), dtype=model.dtype),
    proposal_scale * torch.ones(model.num_params(), dtype=model.dtype)
)
sampler = MetropolisHastings(
    model,
    theta0=model.prior.sample(),
    dataloader=dataloader,
    kernel=kernel
)

# %% Run Metropolis-Hastings sampler

start_time = timer()

sampler.run(num_epochs=11000, num_burnin_epochs=1000)

end_time = timer()
print("Time taken: {}".format(timedelta(seconds=end_time-start_time)))

# %% For convenience, name the chain list

chain_list = sampler.get_chain()

# %% Compute acceptance rate

print('Acceptance rate: {}'.format(chain_list.acceptance_rate()))

# %% Compute Monte Carlo mean

print('Monte Carlo mean: {}'.format(chain_list.mean()))

# %% Compute Monte Carlo standard error

print('Monte Carlo standard error: {}'.format(chain_list.mc_se()))

# %% Compute multivariate ESS

print('Multivariate ESS: {}'.format(chain_list.multi_ess()))

# %% Generate kanga ChainArray from eeyore ChainList

chain_array = chain_list.to_kanga()

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
