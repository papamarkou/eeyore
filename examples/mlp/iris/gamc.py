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
from eeyore.samplers import GAMC
from eeyore.stats import softabs

# %% Avoid issuing memory warning due to number of plots

plt.rcParams.update({'figure.max_open_warning': 0})

# %% Load iris data

iris = XYDataset.from_eeyore('iris', yndmin=1, dtype=torch.float32, yonehot=True)
dataloader = DataLoader(iris, batch_size=len(iris), shuffle=True)

# %% Setup MLP model

# Use torch.float64 to avoid numerical issues associated with eigenvalue computation in softabs
# See https://github.com/pytorch/pytorch/issues/24466

hparams = mlp.Hyperparameters(dims=[4, 3, 3], activations=[torch.sigmoid, None])

model = mlp.MLP(
    loss=loss_functions['multiclass_classification'],
    hparams=hparams,
    dtype=torch.float32
)

model.prior = Normal(
    torch.zeros(model.num_params(), dtype=model.dtype),
    np.sqrt(3)*torch.ones(model.num_params(), dtype=model.dtype)
)

# %% Set initial values of chains

theta0 = model.prior.sample()

# %% Setup GAMC sampler

per_chain_samplers = [
    ['AM', {
        'l': 0.01, 'b': 2., 'c': 0.025,
        'transform': lambda hessian: softabs(hessian.to(torch.float64), 1000.).to(torch.float32)
    }],
    # ['RAM', {}],
    ['SMMALA', {
        'step': 0.1,
        'transform': lambda hessian: softabs(hessian.to(torch.float64), 1000.).to(torch.float32)
    }]
]

sampler = GAMC(model, theta0, dataloader, per_chain_samplers)

# %% Run PowerPosteriorSampler

start_time = timer()

sampler.run(num_epochs=11000, num_burnin_epochs=1000, verbose=True, verbose_step=1000)

end_time = timer()
print("Time taken: {}".format(timedelta(seconds=end_time-start_time)))

# %% Compute Monte Carlo mean

sampler.chain.mean()

# %% Plot traces of simulated Markov chain

for i in range(sampler.samplers[0].model.num_params()):
    chain = sampler.chain.get_sample(i)
    plt.figure()
    sns.lineplot(range(len(chain)), chain)
    plt.xlabel('Iteration')
    plt.ylabel('Parameter value')
    plt.title(r'Traceplot of parameter {}'.format(i+1))

# %% Plot running means of simulated Markov chain

for i in range(sampler.samplers[0].model.num_params()):
    chain = sampler.chain.get_sample(i)
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
    sns.distplot(sampler.chain.get_sample(i), bins=20, norm_hist=True)
    plt.xlabel('Value range')
    plt.ylabel('Relative frequency')
    plt.title(r'Histogram of parameter {}'.format(i+1))
