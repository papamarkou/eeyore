# %% Sampling from a normalized Gamma density via MALA

# %% Import packages

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import seaborn as sns
import torch

from torch.distributions.gamma import Gamma
from torch.utils.data import DataLoader

from eeyore.datasets import EmptyXYDataset
from eeyore.models import DistributionModel
from eeyore.samplers import MALA

# %% Set up unnormalized target density

v = torch.tensor([2., 1.], dtype=torch.float64)

# def log_pdf(theta, x, y):
#     return (v[0] - 1) * theta - torch.exp(theta) / v[1] + theta # Jacobian

def log_pdf(theta, x, y):
    return Gamma(v[0], 1 / v[1]).log_prob(torch.exp(theta)) + theta

model = DistributionModel(log_pdf, 1, dtype=torch.float64)

# %% Setup MALA sampler

sampler = MALA(
    model,
    theta0=torch.tensor([-1], dtype=model.dtype),
    dataloader=DataLoader(EmptyXYDataset()),
    step=0.25
)

# %% Run MALA sampler

sampler.run(num_epochs=11000, num_burnin_epochs=1000)

# %% For convenience, name the chain list

chain_list = sampler.get_chain()

# %% Compute acceptance rate

print('Acceptance rate: {}'.format(chain_list.acceptance_rate()))

# %% Compute Monte Carlo mean

print('Monte Carlo mean: {}'.format(chain_list.mean()))

# %% Compute Monte Carlo standard error

print('Monte Carlo standard error: {}'.format(chain_list.mc_se()))

# %% Plot traces of simulated Markov chain

chain = chain_list.get_param(0)
plt.figure()
sns.lineplot(range(len(chain)), chain)
plt.xlabel('Iteration')
plt.ylabel('Parameter value')
plt.title(r'Traceplot of $\theta_{{{}}}$'.format(1))

# %% Plot histograms of marginals of simulated Markov chain

x_hist_range = np.linspace(0.001, 10, 100)

plt.figure()
plot = sns.distplot(
    chain_list.get_param(0).exp(),
    hist=True, norm_hist=True, kde=False,
    color='blue', label='Simulated'
)
plot.set_xlabel('Parameter value')
plot.set_ylabel('Relative frequency')
plot.set_title(r'Traceplot of $\theta_{{{}}}$'.format(1))
sns.lineplot(x_hist_range, stats.gamma.pdf(x_hist_range, v[0], scale=v[1]), color='red', label='Target')
plot.legend()

# np.log(stats.gamma.pdf(2., 2., scale=1.))
# Gamma(v[0], 1 / v[1]).log_prob(torch.tensor(2.))
