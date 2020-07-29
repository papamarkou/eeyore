# %% DE-MC sampling of MLP weights using iris data
#
# Sampling the weights of a multi-layer perceptron (MLP) using the iris data and DE-MC algorithm.

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
from eeyore.samplers import DEMC

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

# %% Setup DEMC sampler

num_chains = 100

sigmas = [torch.tensor(model.num_params()*[0.0001], dtype=model.dtype) for i in range(num_chains)]
c = [0.01 for _ in range(num_chains)]

sampler = DEMC(
    model,
    [torch.tensor(model.num_params()*[0.0001], dtype=model.dtype) for i in range(num_chains)],
    dataloader,
    theta0=model.prior.sample(),
    num_chains=num_chains,
    c=[0.01 for _ in range(num_chains)]
)

# %% Run DEMC

start_time = timer()

sampler.run(num_epochs=11000, num_burnin_epochs=1000, verbose=True, verbose_step=1000)

end_time = timer()
print("Time taken: {}".format(timedelta(seconds=end_time-start_time)))

# %% Select one of the chains to generate diagnostics

chain_id = 0

# %% Compute acceptance rate

print('Acceptance rate: {}'.format(sampler.get_chain(idx=chain_id).acceptance_rate()))

# %% Compute Monte Carlo mean

print('Monte Carlo mean: {}'.format(sampler.get_chain(idx=chain_id).mean()))

# %% Plot traces of simulated Markov chain

for j in range(sampler.get_model(idx=chain_id).num_params()):
    chain = sampler.get_sample(j, chain_idx=chain_id)
    plt.figure()
    sns.lineplot(range(len(chain)), chain)
    plt.xlabel('Iteration')
    plt.ylabel('Parameter value')
    plt.title(r'Traceplot of parameter {}'.format(j+1))

# %% Plot running means of simulated Markov chain

for j in range(sampler.get_model(idx=chain_id).num_params()):
    chain = sampler.get_sample(j, chain_idx=chain_id)
    chain_mean = torch.empty(len(chain))
    chain_mean[0] = chain[0]
    for k in range(1, len(chain)):
        chain_mean[k] = (chain[j]+j*chain_mean[k-1])/(k+1)

    plt.figure()
    sns.lineplot(range(len(chain)), chain_mean)
    plt.xlabel('Iteration')
    plt.ylabel('Parameter value')
    plt.title(r'Running mean of parameter {}'.format(j+1))

# %% Plot histograms of simulated Markov chain

for j in range(sampler.get_model(idx=chain_id).num_params()):
    plt.figure()
    sns.distplot(sampler.get_sample(j, chain_idx=chain_id), bins=20, norm_hist=True)
    plt.xlabel('Value range')
    plt.ylabel('Relative frequency')
    plt.title(r'Histogram of parameter {}'.format(j+1))
