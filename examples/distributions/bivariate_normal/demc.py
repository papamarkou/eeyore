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

# %% Set up empty data loader

dataloader = DataLoader(EmptyXYDataset())

# for data, label in dataloader:
#     print("Data :", data)
#     print("Label :", label)

# %% Set up unnormalized target density

# Using manually defined log_pdf function

# def log_pdf(theta, x, y):
#     return -0.5*torch.sum(theta**2)

# Using log_pdf function based on Normal torch distribution

# pdf = Normal(torch.zeros(2), torch.ones(2))

# def log_pdf(theta, x, y):
#     return torch.sum(pdf.log_prob(theta))

# Using log_pdf function based on MultivariateNormal torch distribution

pdf = MultivariateNormal(torch.zeros(2), covariance_matrix=torch.eye(2))

def log_pdf(theta, x, y):
    return pdf.log_prob(theta)

density = Density(log_pdf, 2, dtype=torch.float32)

# %% Set initial values of chains

theta0 = torch.tensor([-1, 1], dtype=torch.float32)

# %% Set proposal scale

num_chains = 5

sigmas = [torch.ones(2) for _ in range(num_chains)]

# %% Setup DE-MC sampler

c = [1. for _ in range(num_chains)]
sampler = DEMC(density, theta0, sigmas, dataloader, num_chains=num_chains, c=c)

# %% Run DE-MC sampler

sampler.run(num_epochs=11000, num_burnin_epochs=1000, verbose=True, verbose_step=1000)

# %% Select one of the chains to generate diagnostics

chain_id = 0

# %% Compute acceptance rate

sampler.samplers[chain_id].chain.acceptance_rate()

# %% Compute Monte Carlo mean

sampler.samplers[chain_id].chain.mean()

# %% Plot traces of simulated Markov chain

for i in range(sampler.samplers[chain_id].model.num_params()):
    chain = sampler.samplers[chain_id].chain.get_sample(i)
    plt.figure()
    sns.lineplot(range(len(chain)), chain)
    plt.xlabel('Iteration')
    plt.ylabel('Parameter value')
    plt.title(r'Traceplot of parameter $\theta_{}$'.format(i+1))

# %% Plot histograms of marginals of simulated Markov chain

x_hist_range = np.linspace(-4, 4, 100)

for i in range(sampler.samplers[chain_id].model.num_params()):
    plt.figure()
    plot = sns.distplot(
        sampler.samplers[chain_id].chain.get_sample(i), hist=False, color='blue', label='Simulated'
    )
    plot.set_xlabel('Parameter value')
    plot.set_ylabel('Relative frequency')
    plot.set_title(r'Traceplot of parameter $\theta_{}$'.format(i+1))
    sns.lineplot(x_hist_range, stats.norm.pdf(x_hist_range, 0, 1), color='red', label='Target')
    plot.legend()

# %% Plot scatter of simulated Markov chain

x_contour_range, y_contour_range = np.mgrid[-5:5:.01, -5:5:.01]

contour_grid = np.empty(x_contour_range.shape+(2,))
contour_grid[:, :, 0] = x_contour_range
contour_grid[:, :, 1] = y_contour_range

target = stats.multivariate_normal([0., 0.], [[1., 0.], [0., 1.]])

plt.scatter(
    x=sampler.samplers[chain_id].chain.get_sample(0),
    y=sampler.samplers[chain_id].chain.get_sample(1),
    marker='+'
)
plt.contour(x_contour_range, y_contour_range, target.pdf(contour_grid), cmap='copper')
plt.title('Countours of target and scatterplot of simulated chain');

# %% Plot KDE of target of simulated Markov chain

plot = sns.kdeplot(
    sampler.samplers[chain_id].chain.get_sample(0),
    sampler.samplers[chain_id].chain.get_sample(1),
    shade=True
)
plot.set_title('KDE of simulated chain');

# %% Plot KDEs of target and of marginals of simulated Markov chain

plot = sns.jointplot(
    sampler.samplers[chain_id].chain.get_sample(0),
    sampler.samplers[chain_id].chain.get_sample(1),
    kind="kde"
)

# %% Plot scatter of target and histograms of marginals of simulated Markov chain

sns.jointplot(
    sampler.samplers[chain_id].chain.get_sample(0),
    sampler.samplers[chain_id].chain.get_sample(1),
    kind="scatter"
);
