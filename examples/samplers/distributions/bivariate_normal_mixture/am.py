# %% Sampling from a bivariate normal mixture via AM

# %% Import packages

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import scipy.stats as stats
import torch

# from torch.distributions import MultivariateNormal
from torch.utils.data import DataLoader

from eeyore.datasets import EmptyXYDataset
from eeyore.models import DistributionModel
from eeyore.samplers import AM
from eeyore.stats import softabs

# %% Set up unnormalized target density

# Use torch.float64 to avoid numerical issues associated with eigenvalue computation in softabs
# See https://github.com/pytorch/pytorch/issues/24466

pdf_dtype = torch.float32

# def pdf(theta, w, components):
#     return w[0] * torch.exp(components[0].log_prob(theta)) + w[1] * torch.exp(components[1].log_prob(theta))

means = [-2 * torch.ones(2, dtype=pdf_dtype), 2 * torch.ones(2, dtype=pdf_dtype)]

weights = torch.tensor([0.5, 0.5], dtype=pdf_dtype)
covs = [1 * torch.eye(2, dtype=pdf_dtype), 1 * torch.eye(2, dtype=pdf_dtype)]

# def log_pdf(theta, x, y):
#     return torch.log(pdf(
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

model = DistributionModel(log_pdf, 2, dtype=pdf_dtype)

# %% Setup AM sampler

# softabs is used for avoiding issues with Cholesky decomposition
# See https://github.com/pytorch/pytorch/issues/24466
# Relevant functions :np.linalg.eig(), torch.eig() and torch.symeig()
# If operations are carried out in torch.float64, Cholesky fails
# The solution is to use torch.float32 throught, and convert to torch.float64 only in softabs

sampler = AM(
    model,
    theta0=torch.tensor([0., 0.], dtype=model.dtype),
    dataloader=DataLoader(EmptyXYDataset()),
    l=0.01,
    b=2.,
    c=2.,
    transform=lambda hessian: softabs(hessian.to(torch.float64), 1000.).to(model.dtype)
)

# %% Run AM sampler

sampler.run(num_epochs=11000, num_burnin_epochs=1000)

# %% Compute acceptance rate

print('Acceptance rate: {}'.format(sampler.get_chain().acceptance_rate()))

# %% Compute Monte Carlo mean

print('Monte Carlo mean: {}'.format(sampler.get_chain().mean()))

# %% Compute posterior covariance matrix

mc_cov_mat = sampler.get_chain().mc_cov()

# %% Compute Monte Carlo standard error

print('Monte Carlo standard error: {}'.format(sampler.get_chain().mc_se(mc_cov_mat=mc_cov_mat)))

# %% Compute multivariate ESS

print('Multivariate ESS: {}'.format(sampler.get_chain().multi_ess(mc_cov_mat=mc_cov_mat)))

# %% Plot traces of simulated Markov chain

for i in range(model.num_params()):
    chain = sampler.get_param(i)
    plt.figure()
    sns.lineplot(x=range(len(chain)), y=chain)
    plt.xlabel('Iteration')
    plt.ylabel('Parameter value')
    plt.title(r'Traceplot of $\theta_{{{0}}}$'.format(i+1))

# %% Plot histograms of marginals of simulated Markov chain

x_hist_range = np.linspace(-7, 7, 100)

for i in range(model.num_params()):
    plt.figure()
    plot = sns.kdeplot(sampler.get_param(i), color='blue', label='Simulated')
    plot.set_xlabel('Parameter value')
    plot.set_ylabel('Relative frequency')
    plot.set_title(r'Traceplot of $\theta_{{{0}}}$'.format(i+1))
    sns.lineplot(
        x=x_hist_range,
        y=weights[0] * stats.norm.pdf(x_hist_range, means[0][i].item(), covs[0][i, i]) +
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

plt.scatter(x=sampler.get_param(0), y=sampler.get_param(1), marker='+')
plt.contour(x_contour_range, y_contour_range, target_scipy(contour_grid), cmap='copper')
plt.title('Countours of target and scatterplot of simulated chain');
