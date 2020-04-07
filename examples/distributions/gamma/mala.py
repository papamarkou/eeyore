# %% Sampling from a Gamma density via MALA

# %% Import packages

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import seaborn as sns
import torch

from torch.utils.data import DataLoader

from eeyore.datasets import EmptyXYDataset
from eeyore.models import Density
from eeyore.samplers import MALA

# %% Set up empty data loader

dataloader = DataLoader(EmptyXYDataset())

# for data, label in dataloader:
#     print("Data :", data)
#     print("Label :", label)

# %% Set up unnormalized target density

v = torch.tensor([2., 1.], dtype=torch.float64)

def log_pdf(theta, x, y):
    return (v[0] - 1) * theta - torch.exp(theta) / v[1] + theta # Jacobian

density = Density(log_pdf, 1, dtype=torch.float64)

# %% Setup MALA sampler

theta0 = torch.tensor([-1], dtype=torch.float64)
sampler = MALA(density, theta0, dataloader, step=0.25)

# %% Run MALA sampler

sampler.run(num_epochs=11000, num_burnin_epochs=1000)

# %% Compute acceptance rate

sampler.chain.acceptance_rate()

# %% Compute Monte Carlo mean

sampler.chain.mean()

# %% Plot traces of simulated Markov chain

chain = sampler.chain.get_sample(0)
plt.figure()
sns.lineplot(range(len(chain)), chain)
plt.xlabel('Iteration')
plt.ylabel('Parameter value')
plt.title(r'Traceplot of parameter $\theta_{}$'.format(1))

# %% Plot histograms of marginals of simulated Markov chain

x_hist_range = np.linspace(0.001, 10, 100)

plt.figure()
plot = sns.distplot(
    torch.tensor(sampler.chain.get_sample(0)).exp(),
    hist=True, norm_hist=True, kde=False,
    color='blue', label='Simulated'
)
plot.set_xlabel('Parameter value')
plot.set_ylabel('Relative frequency')
plot.set_title(r'Traceplot of parameter $\theta_{}$'.format(1))
sns.lineplot(x_hist_range, stats.gamma.pdf(x_hist_range, v[0], scale=v[1]), color='red', label='Target')
plot.legend()
