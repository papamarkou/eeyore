# %% SMMALA sampling of MLP weights using XOR data
# 
# Learn the XOR function by sampling the weights of a multi-layer perceptron (MLP) via SMMALA.

# %% Import packages

import matplotlib.pyplot as plt
import seaborn as sns
import torch

from torch.distributions import Normal
from torch.utils.data import DataLoader

from eeyore.constants import loss_functions
from eeyore.datasets import XYDataset
from eeyore.models import mlp
from eeyore.samplers import SMMALA
from eeyore.stats import softabs

# %% Load XOR data

xor = XYDataset.from_eeyore('xor', dtype=torch.float32)
dataloader = DataLoader(xor, batch_size=len(xor))

# %% Setup MLP model

# Use torch.float64 to avoid numerical issues associated with eigenvalue computation in softabs
# See https://github.com/pytorch/pytorch/issues/24466

hparams = mlp.Hyperparameters(dims=[2, 2, 1])
model = mlp.MLP(loss=loss_functions['binary_classification'], hparams=hparams, dtype=torch.float32)
model.prior = Normal(
    torch.zeros(model.num_params(), dtype=model.dtype),
    torch.sqrt(torch.tensor(3, dtype=model.dtype)) * torch.ones(model.num_params(), dtype=model.dtype)
)

# %% Setup SMMALA sampler

# softabs is used for avoiding issues with Cholesky decomposition
# See https://github.com/pytorch/pytorch/issues/24466
# Relevant functions :np.linalg.eig(), torch.eig() and torch.symeig() 
# If operations are carried out in torch.float64, Cholesky fails
# The solution is to use torch.float32 throught, and convert to torch.float64 only in softabs
# However, in this XOR example even such solution seems to not work well with torch.symeig()

theta0 = model.prior.sample()
sampler = SMMALA(
    model, theta0, dataloader,
    step=1.,
    transform=lambda hessian: softabs(hessian.to(torch.float64), 1000.).to(torch.float32)
)

# %% Run SMMALA sampler

sampler.run(num_epochs=5500, num_burnin_epochs=500, verbose=True, verbose_step=500)

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
    plt.title(r'Traceplot of parameter $\theta_{}$'.format(i+1))

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
    plt.title(r'Running mean of parameter $\theta_{}$'.format(i+1))

# %% Plot histograms of simulated Markov chain

for i in range(model.num_params()):
    plt.figure()
    sns.distplot(sampler.chain.get_sample(i), bins=20, norm_hist=True)
    plt.xlabel('Value range')
    plt.ylabel('Relative frequency')
    plt.title(r'Histogram of parameter $\theta_{}$'.format(i+1))
