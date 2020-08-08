# %% SMMALA sampling of logistic regression coefficients using Swiss banknote data
#
# Sampling logistic regression coefficients using the Swiss banknote data and SMMALA.

# %% Import packages

import matplotlib.pyplot as plt
import seaborn as sns
import torch

from datetime import timedelta
from timeit import default_timer as timer
from torch.distributions import Normal
from torch.utils.data import DataLoader

from eeyore.chains import ChainList
from eeyore.datasets import XYDataset
from eeyore.models import logistic_regression
from eeyore.samplers import SMMALA
from eeyore.stats import binary_cross_entropy #, softabs

# %% Load and standardize Swiwss banknote data

banknotes = XYDataset.from_eeyore('banknotes', dtype=torch.float32)
banknotes.x = banknotes.x[: , :4]
banknotes.x = \
    (banknotes.x - torch.mean(banknotes.x, dim=0, keepdim=True))/ \
    torch.std(banknotes.x, dim=0, keepdim=True, unbiased=False)

dataloader = DataLoader(banknotes, batch_size=len(banknotes))

# %% Setup logistic regression model

hparams = logistic_regression.Hyperparameters(input_size=4, bias=False)
model = logistic_regression.LogisticRegression(
    loss=lambda x, y: binary_cross_entropy(x, y, reduction='sum'),
    hparams=hparams,
    dtype=torch.float32
)
model.prior = Normal(
    torch.zeros(model.num_params(), dtype=model.dtype),
    torch.sqrt(torch.tensor(3, dtype=model.dtype)) * torch.ones(model.num_params(), dtype=model.dtype)
)

# %% Setup SMMALA sampler

sampler = SMMALA(
    model,
    theta0=model.prior.sample(),
    dataloader=dataloader,
    step=0.1, # ,
    # transform=lambda hessian: softabs(hessian.to(torch.float64), 1000.).to(torch.float32)
    chain=ChainList(keys=['sample', 'target_val', 'grad_val', 'accepted'])
)

# %% Run SMMALA sampler

start_time = timer()

sampler.run(num_epochs=11000, num_burnin_epochs=1000, verbose=True, verbose_step=1000)

end_time = timer()
print("Time taken: {}".format(timedelta(seconds=end_time-start_time)))

# %%

chain_array = sampler.get_chain().to_kanga()

# %% Compute acceptance rate

print('Acceptance rate: {}'.format(sampler.get_chain().acceptance_rate()))

# %% Compute Monte Carlo mean

print('Monte Carlo mean: {}'.format(sampler.get_chain().mean()))

# %% Plot traces of simulated Markov chain

for i in range(model.num_params()):
    chain = sampler.get_sample(i)
    plt.figure()
    sns.lineplot(range(len(chain)), chain)
    plt.xlabel('Iteration')
    plt.ylabel('Parameter value')
    plt.title(r'Traceplot of $\theta_{{{0}}}$'.format(i+1))

# %% Plot running means of simulated Markov chain

for i in range(model.num_params()):
    chain = sampler.get_sample(i)
    chain_mean = torch.empty(len(chain))
    chain_mean[0] = chain[0]
    for j in range(1, len(chain)):
        chain_mean[j] = (chain[j]+j*chain_mean[j-1])/(j+1)

    plt.figure()
    sns.lineplot(range(len(chain)), chain_mean)
    plt.xlabel('Iteration')
    plt.ylabel('Parameter value')
    plt.title(r'Running mean of $\theta_{{{0}}}$'.format(i+1))

# %% Plot histograms of marginals of simulated Markov chain

for i in range(model.num_params()):
    plt.figure()
    sns.distplot(sampler.get_sample(i), bins=20, norm_hist=True)
    plt.xlabel('Value range')
    plt.ylabel('Relative frequency')
    plt.title(r'Histogram of $\theta_{{{0}}}$'.format(i+1))
