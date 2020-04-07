# %% Power posterior sampling of MLP weights using iris data
# 
# Sampling the weights of a multi-layer perceptron (MLP) using the iris data and power posterior algorithm.

# %% Import packages

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch

from datetime import timedelta
from timeit import default_timer as timer
from torch.distributions import Normal
from torch.utils.data import DataLoader

from eeyore.constants import loss_functions
from eeyore.datasets import XYDataset
from eeyore.models import mlp
from eeyore.samplers import PowerPosteriorSampler

# %% Avoid issuing memory warning due to number of plots

plt.rcParams.update({'figure.max_open_warning': 0})

# %% Load iris data

iris = XYDataset.from_eeyore('iris', yndmin=1, yonehot=True)
dataloader = DataLoader(iris, batch_size=len(iris), shuffle=True)

# %% Setup MLP model

hparams = mlp.Hyperparameters(dims=[4, 3, 3], activations=[torch.sigmoid, None])

model = mlp.MLP(
    loss=loss_functions['multiclass_classification'],
    hparams=hparams,
    dtype=torch.float64
)

model.prior = Normal(
    torch.zeros(model.num_params(), dtype=model.dtype),
    np.sqrt(3)*torch.ones(model.num_params(), dtype=model.dtype)
)

# %% Set initial values of chains

theta0 = model.prior.sample()

# %% Setup PowerPosteriorSampler

num_chains = 10

drift_step = 0.0023
per_chain_samplers = [['MALA', {'step': drift_step}] for _ in range(num_chains)]
sampler = PowerPosteriorSampler(
    model, theta0, dataloader, per_chain_samplers, between_step=1, temperature=[1 for _ in range(num_chains)]
)

# %% Run PowerPosteriorSampler

start_time = timer()

sampler.run(num_epochs=11000, num_burnin_epochs=1000, verbose=True, verbose_step=100)

end_time = timer()
print("Time taken: {}".format(timedelta(seconds=end_time-start_time)))

# %% Compute Monte Carlo mean

sampler.get_chain().mean()

# %% Plot traces of simulated Markov chain

for i in range(sampler.samplers[0].model.num_params()):
    chain = sampler.get_chain().get_sample(i)
    plt.figure()
    sns.lineplot(range(len(chain)), chain)
    plt.xlabel('Iteration')
    plt.ylabel('Parameter value')
    plt.title(r'Traceplot of parameter {}'.format(i+1))

# %% Plot running means of simulated Markov chain

for i in range(sampler.samplers[0].model.num_params()):
    chain = sampler.get_chain().get_sample(i)
    chain_mean = torch.empty(len(chain))
    chain_mean[0] = chain[0]
    for j in range(1, len(chain)):
        chain_mean[j] = (chain[j]+j*chain_mean[j-1])/(j+1)
        
    plt.figure()
    sns.lineplot(range(len(chain)), chain_mean)
    plt.xlabel('Iteration')
    plt.ylabel('Parameter value')
    plt.title(r'Running mean of parameter {}'.format(i+1))

# %% Plot histograms of simulated Markov chain

for i in range(sampler.samplers[0].model.num_params()):
    plt.figure()
    sns.distplot(sampler.get_chain().get_sample(i), bins=20, norm_hist=True)
    plt.xlabel('Value range')
    plt.ylabel('Relative frequency')
    plt.title(r'Histogram of parameter {}'.format(i+1))
