import os
import copy

import numpy as np

from timeit import default_timer as timer
from datetime import timedelta

import torch

from eeyore.stats import choose_from_subset
from eeyore.api import Sampler
from eeyore.kernels import DEMCKernel
from eeyore.mcmc import ChainFile, ChainList
from .metropolis_hastings import MetropolisHastings

class DEMC(Sampler):
    def __init__(self, models, theta0s, dataloaders, sigmas, num_chains=10, c=None, schedule=lambda n,
    num_iterations: 1., storage='list', keys=['theta', 'target_val', 'accepted'], path=os.getcwd(), mode='a'):
        super(DEMC, self).__init__()
        self.models = models
        self.dataloaders = dataloaders
        self.num_chains = num_chains
        self.sigmas = sigmas
        self.schedule = schedule
        self.temperature = None

        self.c = c or [2.38/np.sqrt(self.models[i].num_params()) for i in range(self.num_chains)]

        self.kernels = [DEMCKernel(c=self.c[i]) for i in range(self.num_chains)]
        for i in range(self.num_chains):
            self.kernels[i].init_a_and_b(self.models[i].num_params(), self.models[i].dtype, self.models[i].device)
            self.kernels[i].init_density(self.models[i].num_params(), self.models[i].dtype, self.models[i].device)
            self.kernels[i].density.scale = self.sigmas[i]

        self.chains = []
        for i in range(self.num_chains):
            if storage == 'list':
                self.chains.append(ChainList(keys=keys))
            elif storage == 'file':
                chain_path = os.path.join(path, 'chain'+f"{(i+1):0{len(str(self.num_chains))}}")
                if not os.path.exists(chain_path):
                    os.makedirs(chain_path)
                self.chains.append(ChainFile(keys=keys, path=chain_path, mode=mode))

        self.samplers = []
        for i in range(self.num_chains):
            self.samplers.append(MetropolisHastings(
                self.models[i], theta0s[i], self.dataloaders[i],
                symmetric=True, kernel=self.kernels[i], chain=self.chains[i]
            ))

    def set_temperature(self, n, num_iterations):
        self.temperature = self.schedule(n, num_iterations)
        for i in range(self.num_chains):
            self.models[i].temperature = self.temperature

    def set_kernel(self, i):
        j = choose_from_subset(self.num_chains, [i])
        k = choose_from_subset(self.num_chains, [i, j])
        self.samplers[i].kernel.set_a_and_b(
            self.samplers[j].current['theta'].clone().detach(), self.samplers[k].current['theta'].clone().detach()
        )
        self.samplers[i].kernel.set_density_params(self.samplers[i].current['theta'].clone().detach())

    def draw(self, n, savestate=False):
        for i in range(self.num_chains):
            self.set_kernel(i)
            self.samplers[i].draw(n, savestate=savestate)

    def run(self, num_iterations, num_burnin, verbose=False, verbose_step=100):
        verbose_msg = "Iteration {:" + str(len(str(num_iterations))) + "}, duration {}"

        for n in range(num_iterations):
            if verbose and (((n+1) % verbose_step) == 0):
                start_time = timer()

            savestate = False if (n < num_burnin) else True

            self.set_temperature(n, num_iterations)

            self.draw(n, savestate=savestate)

            if verbose and (((n+1) % verbose_step) == 0):
                end_time = timer()
                print(verbose_msg.format(n+1, timedelta(seconds=end_time-start_time)))
