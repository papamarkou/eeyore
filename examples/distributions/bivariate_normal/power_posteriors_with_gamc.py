# %% Sampling from a bivariate normal density via power posterior sampler

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
from eeyore.models import Density
from eeyore.samplers import PowerPosteriorSampler
from eeyore.stats import softabs

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

# %% Setup PowerPosteriorSampler

num_chains = 5

per_gamc_chain_samplers = [
    # ['AM', {
    #     'l': 0.005, 'b': 4., 'c': 4.,
    #     'transform': lambda hessian: softabs(hessian.to(torch.float64), 1000.).to(torch.float32)
    # }],
    ['RAM', {}],
    ['SMMALA', {
        'step': 0.25,
        'transform': lambda hessian: softabs(hessian.to(torch.float64), 1000.).to(torch.float32)
    }]
]

per_pp_chain_samplers = [['GAMC', {'samplers': per_gamc_chain_samplers}] for _ in range(num_chains)]

sampler = PowerPosteriorSampler(density, theta0, dataloader, per_pp_chain_samplers, between_step=1, check_input=True)

# %% Run PowerPosteriorSampler

start_time = timer()

sampler.run(num_epochs=11000, num_burnin_epochs=1000, verbose=True, verbose_step=1000)

end_time = timer()
print("Time taken: {}".format(timedelta(seconds=end_time-start_time)))

# %% Compute Monte Carlo mean

sampler.get_chain().mean()

# %% Plot traces of simulated Markov chain

for i in range(sampler.samplers[0].samplers[0].model.num_params()):
    chain = sampler.get_chain().get_sample(i)
    plt.figure()
    sns.lineplot(range(len(chain)), chain)
    plt.xlabel('Iteration')
    plt.ylabel('Parameter value')
    plt.title(r'Traceplot of parameter $\theta_{}$'.format(i+1))

# %% Plot histograms of marginals of simulated Markov chain

x_hist_range = np.linspace(-4, 4, 100)

for i in range(sampler.samplers[0].samplers[0].model.num_params()):
    plt.figure()
    plot = sns.distplot(sampler.get_chain().get_sample(i), hist=False, color='blue', label='Simulated')
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

plt.scatter(x=sampler.get_chain().get_sample(0), y=sampler.get_chain().get_sample(1), marker='+')
plt.contour(x_contour_range, y_contour_range, target.pdf(contour_grid), cmap='copper')
plt.title('Countours of target and scatterplot of simulated chain');

# %% Plot KDE of target of simulated Markov chain

plot = sns.kdeplot(sampler.get_chain().get_sample(0), sampler.get_chain().get_sample(1), shade=True)
plot.set_title('KDE of simulated chain');

# %% Plot KDEs of target and of marginals of simulated Markov chain

plot = sns.jointplot(sampler.get_chain().get_sample(0), sampler.get_chain().get_sample(1), kind="kde")

# %% Plot scatter of target and histograms of marginals of simulated Markov chain

sns.jointplot(sampler.get_chain().get_sample(0), sampler.get_chain().get_sample(1), kind="scatter");
