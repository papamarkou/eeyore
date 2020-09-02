# %% Sampling from a bivariate normal density via DE-MC

# %% Import packages

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import seaborn as sns
import torch

# from torch.distributions import Normal
from torch.distributions import MultivariateNormal
from torch.utils.data import DataLoader

from eeyore.datasets import EmptyXYDataset
from eeyore.models import Density
from eeyore.samplers import DEMC

# %% Set up unnormalized target density

# Using manually defined log_pdf function

# def log_pdf(theta, x, y):
#     return -0.5*torch.sum(theta**2)

# Using log_pdf function based on Normal torch distribution

# pdf = Normal(torch.zeros(2), torch.ones(2))

# def log_pdf(theta, x, y):
#     return torch.sum(pdf.log_prob(theta))

# Using log_pdf function based on MultivariateNormal torch distribution

pdf_dtype = torch.float32

pdf = MultivariateNormal(torch.zeros(2, dtype=pdf_dtype), covariance_matrix=torch.eye(2, dtype=pdf_dtype))

def log_pdf(theta, x, y):
    return pdf.log_prob(theta)

density = Density(log_pdf, 2, dtype=pdf.loc.dtype)

# %% Set number of chains

num_chains = 5

# %% Setup DE-MC sampler

sampler = DEMC(
    density,
    [torch.ones(2) for _ in range(num_chains)],
    DataLoader(EmptyXYDataset()),
    theta0=torch.tensor([-1, 1], dtype=density.dtype),
    num_chains=num_chains,
    c=[1. for _ in range(num_chains)]
)

# %% Run DE-MC sampler

sampler.run(num_epochs=11000, num_burnin_epochs=1000, verbose=True, verbose_step=1000)

# %% Select one of the chains to generate diagnostics

chain_id = 0

# %% Compute acceptance rate

print('Acceptance rate: {}'.format(sampler.get_chain(idx=chain_id).acceptance_rate()))

# %% Compute Monte Carlo mean

print('Monte Carlo mean: {}'.format(sampler.get_chain(idx=chain_id).mean()))

# %% Compute Monte Carlo standard error

print('Monte Carlo standard error: {}'.format(sampler.get_chain(idx=chain_id).mc_se()))

# %% Compute multivariate ESS

print('Multivariate ESS: {}'.format(sampler.get_chain(idx=chain_id).multi_ess()))

# %% Plot traces of simulated Markov chain

for j in range(sampler.get_model(idx=chain_id).num_params()):
    chain = sampler.get_param(j, chain_idx=chain_id)
    plt.figure()
    sns.lineplot(range(len(chain)), chain)
    plt.xlabel('Iteration')
    plt.ylabel('Parameter value')
    plt.title(r'Traceplot of $\theta_{{{0}}}$'.format(j+1))

# %% Plot histograms of marginals of simulated Markov chain

x_hist_range = np.linspace(-4, 4, 100)

for j in range(sampler.get_model(idx=chain_id).num_params()):
    plt.figure()
    plot = sns.distplot(sampler.get_param(j, chain_idx=chain_id), hist=False, color='blue', label='Simulated')
    plot.set_xlabel('Parameter value')
    plot.set_ylabel('Relative frequency')
    plot.set_title(r'Traceplot of $\theta_{{{0}}}$'.format(j+1))
    sns.lineplot(x_hist_range, stats.norm.pdf(x_hist_range, 0, 1), color='red', label='Target')
    plot.legend()

# %% Plot scatter of simulated Markov chain

x_contour_range, y_contour_range = np.mgrid[-5:5:.01, -5:5:.01]

contour_grid = np.empty(x_contour_range.shape+(2,))
contour_grid[:, :, 0] = x_contour_range
contour_grid[:, :, 1] = y_contour_range

target = stats.multivariate_normal([0., 0.], [[1., 0.], [0., 1.]])

plt.scatter(x=sampler.get_param(0, chain_idx=chain_id), y=sampler.get_param(1, chain_idx=chain_id), marker='+')
plt.contour(x_contour_range, y_contour_range, target.pdf(contour_grid), cmap='copper')
plt.title('Countours of target and scatterplot of simulated chain');
