# %% SMMALA sampling of MLP weights using iris data
#
# Sampling the weights of a multi-layer perceptron (MLP) using the iris data and SMMALA.

# %% Import packages for MCMC simulation

import matplotlib.pyplot as plt
import torch

from datetime import timedelta
from timeit import default_timer as timer
from torch.distributions import Normal
from torch.utils.data import DataLoader

from eeyore.constants import loss_functions
from eeyore.datasets import XYDataset
from eeyore.models import mlp
from eeyore.samplers import SMMALA
from eeyore.stats import softabs

# %% Set device to be GPU

device = 'cuda:0'

# %% Avoid issuing memory warning due to number of plots

plt.rcParams.update({'figure.max_open_warning': 0})

# %% Load iris data

iris = XYDataset.from_eeyore('iris', yndmin=1, dtype=torch.float32, device=device, yonehot=True)
dataloader = DataLoader(iris, batch_size=len(iris), shuffle=True)

# %% Setup MLP model

# Use torch.float64 to avoid numerical issues associated with eigenvalue computation in softabs
# See https://github.com/pytorch/pytorch/issues/24466

hparams = mlp.Hyperparameters(dims=[4, 3, 3], activations=[torch.sigmoid, None])
model = mlp.MLP(
    loss=loss_functions['multiclass_classification'],
    hparams=hparams,
    dtype=torch.float32,
    device=device
)
model.prior = Normal(
    torch.zeros(model.num_params(), dtype=model.dtype, device=model.device),
    (3 * torch.ones(model.num_params(), dtype=model.dtype, device=model.device)).sqrt()
)

# %% Setup SMMALA sampler

# softabs is used for avoiding issues with Cholesky decomposition
# See https://github.com/pytorch/pytorch/issues/24466
# Relevant functions :np.linalg.eig(), torch.eig() and torch.symeig()
# If operations are carried out in torch.float64, Cholesky fails
# The solution is to use torch.float32 throught, and convert to torch.float64 only in softabs

sampler = SMMALA(
    model,
    theta0=model.prior.sample(),
    dataloader=dataloader,
    step=0.01,
    transform=lambda hessian:
        softabs(hessian.to(dtype=torch.float64, device=device), 1000.).to(dtype=torch.float32, device=device)
)

# %% Run SMMALA sampler

start_time = timer()

sampler.run(num_epochs=11000, num_burnin_epochs=1000, verbose=True, verbose_step=1000)

end_time = timer()
print("Time taken: {}".format(timedelta(seconds=end_time-start_time)))

# %% Import kanga package for visual MCMC summaries

import kanga.plots as ps

# %% Generate kanga ChainArray from eeyore ChainList

chain_array = sampler.get_chain().to_kanga()

# %% Compute acceptance rate

print('Acceptance rate: {}'.format(chain_array.acceptance_rate()))

# %% Compute Monte Carlo mean

print('Monte Carlo mean: {}'.format(chain_array.mean()))

# %% Compute Monte Carlo standard error

print('Monte Carlo standard error: {}'.format(chain_array.mc_se()))

# %% Compute multivariate ESS

print('Multivariate ESS: {}'.format(chain_array.multi_ess()))

# %% Plot traces of simulated Markov chain

for i in range(model.num_params()):
    ps.trace(
        chain_array.get_param(i),
        title=r'Traceplot of $\theta_{{{}}}$'.format(i+1),
        xlabel='Iteration',
        ylabel='Parameter value'
    )

# %% Plot running means of simulated Markov chain

for i in range(model.num_params()):
    ps.running_mean(
        chain_array.get_param(i),
        title=r'Running mean plot of parameter $\theta_{{{}}}$'.format(i+1),
        xlabel='Iteration',
        ylabel='Running mean'
    )

# %% Plot histograms of marginals of simulated Markov chain

for i in range(model.num_params()):    
    ps.hist(
        chain_array.get_param(i),
        bins=30,
        density=True,
        title=r'Histogram of parameter $\theta_{{{}}}$'.format(i+1),
        xlabel='Parameter value',
        ylabel='Parameter relative frequency'
    )
