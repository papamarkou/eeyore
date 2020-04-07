# %% MALA sampling of MLP weights using iris data
# 
# Sampling the weights of a multi-layer perceptron (MLP) using the iris data and MALA.

# %% Import packages

import matplotlib.pyplot as plt
import seaborn as sns
import torch

from datetime import timedelta
from timeit import default_timer as timer
from torch.distributions import Normal
from torch.utils.data import DataLoader

from eeyore.constants import loss_functions
from eeyore.datasets import XYDataset, XYIDataset
from eeyore.models import mlp
from eeyore.samplers import MALA

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
    torch.sqrt(torch.tensor(3, dtype=model.dtype)) * torch.ones(model.num_params(), dtype=model.dtype)
)

# %% Setup MALA sampler

theta0 = model.prior.sample()
sampler = MALA(model, theta0, dataloader, step=0.0023)

# %% Run MALA sampler

start_time = timer()

sampler.run(num_epochs=11000, num_burnin_epochs=1000)

end_time = timer()
print("Time taken: {}".format(timedelta(seconds=end_time-start_time)))

# %% Compute acceptance rate

sampler.chain.acceptance_rate()

# %% Compute Monte Carlo mean

sampler.chain.mean()

# %% Plot traces of simulated Markov chain

for i in range(model.num_params()):
    chain = sampler.chain.get_sample(i)
    plt.figure()
    sns.lineplot(range(len(chain)), chain)
    plt.xlabel('Iteration')
    plt.ylabel('Parameter value')
    plt.title(r'Traceplot of parameter {}'.format(i+1))

# %% Plot running means of simulated Markov chain

for i in range(model.num_params()):
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

for i in range(model.num_params()):
    plt.figure()
    sns.distplot(sampler.chain.get_sample(i), bins=20, norm_hist=True)
    plt.xlabel('Value range')
    plt.ylabel('Relative frequency')
    plt.title(r'Histogram of parameter {}'.format(i+1))

# %% Compute Monte Carlo approximation of posterior predictive distribution on sampled data points

start_time = timer()

predictive_samples, indices = model.predictive_posterior_from_dataset(
    sampler.chain.vals['sample'],
    XYIDataset.from_xydataset(iris),
    10,
    verbose=True,
    verbose_step=2
)

end_time = timer()
print("Time taken: {}".format(timedelta(seconds=end_time-start_time)))
print("Predictive samples: ", predictive_samples)
print("Indices: ", indices)
