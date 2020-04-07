# %% Sampling from a bivariate normal mixture via RAM

# %% Import packages

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import seaborn as sns
import torch

# from torch.distributions import MultivariateNormal
from torch.utils.data import DataLoader

from eeyore.datasets import EmptyXYDataset
from eeyore.models import Density
from eeyore.samplers import RAM

# %% Set up empty data loader

dataset = EmptyXYDataset()
dataloader = DataLoader(dataset)

# for data, label in dataloader:
#     print("Data :", data)
#     print("Label :", label)

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

# %% Setup RAM sampler

theta0 = torch.tensor([0., 0.], dtype=torch.float32)
density.set_params(theta0)
sampler = RAM(density, theta0, dataloader)

# %% Run RAM sampler

sampler.run(num_epochs=11000, num_burnin_epochs=1000)

# %% Compute acceptance rate

sampler.chain.acceptance_rate()

# %% Compute Monte Carlo mean

sampler.chain.mean()

# %% Plot traces of simulated Markov chain

for i in range(density.num_params()):
    chain = sampler.chain.get_sample(i)
    plt.figure()
    sns.lineplot(range(len(chain)), chain)
    plt.xlabel('Iteration')
    plt.ylabel('Parameter value')
    plt.title(r'Traceplot of parameter $\theta_{}$'.format(i+1))

# %% Plot histograms of marginals of simulated Markov chain

for i in range(density.num_params()):
    range_min = min([means[j][i].item() for j in range(2)])
    range_max = max([means[j][i].item() for j in range(2)])
    range_len = range_max - range_min
    hist_range = np.linspace(range_min - 0.95 * abs(range_len), range_max + 0.95 * abs(range_len), 100)
    plt.figure()
    plot = sns.distplot(sampler.chain.get_sample(i), hist=False, color='blue', label='Simulated')
    plot.set_xlabel('Parameter value')
    plot.set_ylabel('Relative frequency')
    plot.set_title(r'Traceplot of parameter $\theta_{}$'.format(i+1))
    sns.lineplot(
        hist_range,
        weights[0] * stats.norm.pdf(hist_range, means[0][i].item(), covs[0][i, i]) +
        weights[1] * stats.norm.pdf(hist_range, means[1][i].item(), covs[1][i, i]),
        color='red',
        label='Target'
    )
    plot.legend()

# %% Plot scatter of simulated Markov chain

xmin = min([means[j][0].item() for j in range(2)])
xmax = max([means[j][0].item() for j in range(2)])
xlen = xmax - xmin

ymin = min([means[j][1].item() for j in range(2)])
ymax = max([means[j][1].item() for j in range(2)])
ylen = ymax - ymin

x_contour_range, y_contour_range = np.mgrid[
    (xmin - 0.95 * abs(xlen)):(xmax + 0.95 * abs(xlen)):.01,
    (ymin - 0.95 * abs(ylen)):(ymax + 0.95 * abs(ylen)):.01
]

contour_grid = np.empty(x_contour_range.shape+(2,))
contour_grid[:, :, 0] = x_contour_range
contour_grid[:, :, 1] = y_contour_range

# target = stats.multivariate_normal([0., 0.], [[1., 0.], [0., 1.]])

def target_scipy(theta):
    return (
        weights[0] * stats.multivariate_normal(means[0].cpu().numpy(), covs[0].cpu().numpy()).pdf(theta) +
        weights[1] * stats.multivariate_normal(means[1].cpu().numpy(), covs[1].cpu().numpy()).pdf(theta)
    )

plt.scatter(x=sampler.chain.get_sample(0), y=sampler.chain.get_sample(1), marker='+')
plt.contour(x_contour_range, y_contour_range, target_scipy(contour_grid), cmap='copper')
plt.title('Countours of target and scatterplot of simulated chain');

# %% Plot KDE of target of simulated Markov chain

plot = sns.kdeplot(sampler.chain.get_sample(0), sampler.chain.get_sample(1), shade=True)
plot.set_title('KDE of simulated chain');

# %% Plot KDEs of target and of marginals of simulated Markov chain

plot = sns.jointplot(sampler.chain.get_sample(0), sampler.chain.get_sample(1), kind="kde")

# %% Plot scatter of target and histograms of marginals of simulated Markov chain

sns.jointplot(sampler.chain.get_sample(0), sampler.chain.get_sample(1), kind="scatter");
