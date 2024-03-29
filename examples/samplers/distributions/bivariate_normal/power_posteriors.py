# Sampling from a bivariate normal density via power posterior sampler

# %% Import packages

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import seaborn as sns
import torch

from datetime import timedelta
from timeit import default_timer as timer
from torch.distributions import MultivariateNormal
# from torch.distributions import Normal
from torch.utils.data import DataLoader

from eeyore.datasets import EmptyXYDataset
from eeyore.models import DistributionModel
from eeyore.samplers import PowerPosteriorSampler

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

model = DistributionModel(log_pdf, 2, dtype=pdf.loc.dtype)

# %% Setup PowerPosteriorSampler

num_chains = 5

per_chain_samplers = [['MALA', {'step': 0.25}] for _ in range(num_chains)]

sampler = PowerPosteriorSampler(
    model,
    DataLoader(EmptyXYDataset()),
    per_chain_samplers,
    theta0=torch.tensor([-1, 1], dtype=model.dtype),
    between_step=1,
    check_input=True
)

# %% Run PowerPosteriorSampler

start_time = timer()

sampler.run(num_epochs=11000, num_burnin_epochs=1000, verbose=True, verbose_step=1000)

end_time = timer()
print("Time taken: {}".format(timedelta(seconds=end_time-start_time)))

# %% Compute Monte Carlo mean

print('Monte Carlo mean: {}'.format(sampler.get_chain().mean()))

# %% Compute Monte Carlo standard error

print('Monte Carlo standard error: {}'.format(sampler.get_chain().mc_se()))

# %% Compute multivariate ESS

print('Multivariate ESS: {}'.format(sampler.get_chain().multi_ess()))

# %% Plot traces of simulated Markov chain

for i in range(model.num_params()):
    chain = sampler.get_param(i)
    plt.figure()
    sns.lineplot(x=range(len(chain)), y=chain)
    plt.xlabel('Iteration')
    plt.ylabel('Parameter value')
    plt.title(r'Traceplot of $\theta_{{{0}}}$'.format(i+1))

# %% Plot histograms of marginals of simulated Markov chain

x_hist_range = np.linspace(-4, 4, 100)

for i in range(model.num_params()):
    plt.figure()
    plot = sns.kdeplot(sampler.get_param(i), color='blue', label='Simulated')
    plot.set_xlabel('Parameter value')
    plot.set_ylabel('Relative frequency')
    plot.set_title(r'Traceplot of $\theta_{{{0}}}$'.format(i+1))
    sns.lineplot(x=x_hist_range, y=stats.norm.pdf(x_hist_range, 0, 1), color='red', label='Target')
    plot.legend()

# %% Plot scatter of simulated Markov chain

x_contour_range, y_contour_range = np.mgrid[-5:5:.01, -5:5:.01]

contour_grid = np.empty(x_contour_range.shape+(2,))
contour_grid[:, :, 0] = x_contour_range
contour_grid[:, :, 1] = y_contour_range

target = stats.multivariate_normal([0., 0.], [[1., 0.], [0., 1.]])

plt.scatter(x=sampler.get_param(0), y=sampler.get_param(1), marker='+')
plt.contour(x_contour_range, y_contour_range, target.pdf(contour_grid), cmap='copper')
plt.title('Countours of target and scatterplot of simulated chain');
