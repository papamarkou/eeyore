# %% Sampling from a bivariate normal mixture via Metropolis-Hastings

# %% Import packages

import matplotlib.pyplot as plt
import multiprocessing as mp
import numpy as np
import scipy.stats as stats
import seaborn as sns
import torch

from datetime import timedelta
from timeit import default_timer as timer
from torch.distributions import MultivariateNormal
from torch.utils.data import DataLoader

from eeyore.datasets import EmptyXYDataset
# from eeyore.kernels import NormalKernel
from eeyore.kernels import IsoSEKernel, MultivariateNormalKernel
from eeyore.models import Density
from eeyore.samplers import MetropolisHastings
from eeyore.stats import mmd

# %% Set up unnormalized target density

# def pdf(theta, w, components):
#     return w[0] * torch.exp(components[0].log_prob(theta)) + w[1] * torch.exp(components[1].log_prob(theta))

means = [-2 * torch.ones(2), 2 * torch.ones(2)]

# weights and covs are used in plot generation
weights = [0.5, 0.5]
covs = [1 * torch.eye(2), 1 * torch.eye(2)]

# def log_pdf(theta, x, y):
#     return torch.log(target(
#         theta,
#         weights,
#         [
#             MultivariateNormal(means[0], covariance_matrix=covs[0]),
#             MultivariateNormal(means[1], covariance_matrix=covs[1])
#         ]))

def log_pdf(theta, x, y):
    return torch.log(
        torch.exp(-0.5 * (torch.dot(theta-means[0], theta-means[0])))
        + torch.exp(-0.5 * torch.dot(theta-means[1], theta-means[1]))
    )

density = Density(log_pdf, 2, dtype=torch.float32)

# %% Setup Metropolis-Hastings sampler

# kernel = NormalKernel(torch.zeros(2, dtype=torch.float32), torch.ones(2, dtype=torch.float32))
kernel = MultivariateNormalKernel(torch.zeros(2, dtype=torch.float32), torch.eye(2, dtype=torch.float32))
sampler = MetropolisHastings(
    density,
    theta0=torch.tensor([0., 0.], dtype=torch.float32),
    dataloader=DataLoader(EmptyXYDataset()),
    symmetric=True,
    kernel=kernel
)

# %% Run Metropolis-Hastings sampler

sampler.run(num_epochs=11000, num_burnin_epochs=1000)

# %% Compute acceptance rate

print('Acceptance rate: {}'.format(sampler.get_chain().acceptance_rate()))

# %% Compute Monte Carlo mean

print('Monte Carlo mean: {}'.format(sampler.get_chain().mean()))

# %% Plot traces of simulated Markov chain

for i in range(density.num_params()):
    chain = sampler.get_sample(i)
    plt.figure()
    sns.lineplot(range(len(chain)), chain)
    plt.xlabel('Iteration')
    plt.ylabel('Parameter value')
    plt.title(r'Traceplot of $\theta_{{{0}}}$'.format(i+1))

# %% Plot histograms of marginals of simulated Markov chain

x_hist_range = np.linspace(-7, 7, 100)

for i in range(density.num_params()):
    plt.figure()
    plot = sns.distplot(sampler.get_sample(i), hist=False, color='blue', label='Simulated')
    plot.set_xlabel('Parameter value')
    plot.set_ylabel('Relative frequency')
    plot.set_title(r'Traceplot of $\theta_{{{0}}}$'.format(i+1))
    sns.lineplot(
        x_hist_range,
        weights[0] * stats.norm.pdf(x_hist_range, means[0][i].item(), covs[0][i, i]) +
        weights[1] * stats.norm.pdf(x_hist_range, means[1][i].item(), covs[1][i, i]),
        color='red',
        label='Target'
    )
    plot.legend()

# %% Plot scatter of simulated Markov chain

x_contour_range, y_contour_range = np.mgrid[-5:5:.01, -5:5:.01]

contour_grid = np.empty(x_contour_range.shape+(2,))
contour_grid[:, :, 0] = x_contour_range
contour_grid[:, :, 1] = y_contour_range

# target = stats.multivariate_normal([0., 0.], [[1., 0.], [0., 1.]])

def target_scipy(theta):
    return (
        weights[0] * stats.multivariate_normal(means[0].cpu().numpy(), covs[0].cpu().numpy()).pdf(theta) +
        weights[1] * stats.multivariate_normal(means[1].cpu().numpy(), covs[1].cpu().numpy()).pdf(theta)
    )

plt.scatter(x=sampler.get_sample(0), y=sampler.get_sample(1), marker='+')
plt.contour(x_contour_range, y_contour_range, target_scipy(contour_grid), cmap='copper')
plt.title('Countours of target and scatterplot of simulated chain');

# %% Sample directly from mixture

num_samples = list(range(2, 101))
# num_samples = list(range(2, 10)) + list(range(10, 110, 10))
num_samples_max = max(num_samples)

def sample_mixture(n):
    samples = []

    for i in range(n):
        w = torch.rand(1)
        if w < weights[0]:
            samples.append(MultivariateNormal(means[0], covariance_matrix=covs[0]).sample())
        else:
            samples.append(MultivariateNormal(means[1], covariance_matrix=covs[1]).sample())

    return samples

mixture_sample = sample_mixture(num_samples_max)

# %% Compute MMD between MCMC samples and samples generated directly from mixture using multiprocessing

# Step 1: Initialize multiprocessing pool

pool = mp.Pool(mp.cpu_count())
# pool = mp.Pool(4)

# Step 2: `pool.starmap` the `mmd` function

args = [(sampler.chain.vals['sample'][0:n], mixture_sample[0:n], IsoSEKernel()) for n in num_samples]

start_time = timer()

mmd_vals_mp = pool.starmap(mmd, args)

end_time = timer()
print("Time taken to compute MMD with multiprocessing: {}".format(timedelta(seconds=end_time-start_time)))

# Step 3: Close the pool
pool.close()

mmd_vals_mp = [mmd_val_mp.item() for mmd_val_mp in mmd_vals_mp]

# Plot MMD between MCMC samples and samples generated directly from mixture

plot = sns.lineplot(num_samples, mmd_vals_mp)
plot.set_title('MMD btwn MCMC samples and sample() - multiprocessing');

# %% Compute MMD between MCMC samples and samples generated directly from mixture without using multiprocessing

mmd_vals = []

start_time = timer()

for n in num_samples:
    mmd_vals.append(mmd(sampler.chain.vals['sample'][0:n], mixture_sample[0:n], IsoSEKernel()).item())

end_time = timer()
print("Time taken to compute MMD without multiprocessing: {}".format(timedelta(seconds=end_time-start_time)))

# Plot MMD between MCMC samples and samples generated directly from mixture

plot = sns.lineplot(num_samples, mmd_vals)
plot.set_title('MMD btwn MCMC samples and sample() - serial');
