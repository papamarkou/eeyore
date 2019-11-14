import os
import copy

import torch

from eeyore.stats import choose_from_subset
from eeyore.api import Sampler
from eeyore.kernels import DEMCKernel
from eeyore.mcmc import ChainFile, ChainList
from .metropolis_hastings import MetropolisHastings

class DEMC(Sampler):
    def __init__(self, model, theta0, dataloader, sigma, num_chains=10, c=0.1, schedule=lambda n, num_iterations: 1.,
    storage='list', keys=['theta', 'target_val', 'accepted'], path=os.getcwd(), mode='a'):
        super(DEMC, self).__init__()
        self.num_chains = num_chains
        self.sigma = sigma
        self.c = c
        self.schedule = schedule
        self.temperature = None

        self.models = []
        for i in range(self.num_chains):
            self.models.append(copy.deepcopy(model))

        # Define chains
        self.chains = []
        for i in range(self.num_chains):
            if storage == 'list':
                self.chains.append(ChainList(keys=keys))
            elif storage == 'file':
                chain_path = os.path.join(path, 'chain'+f"{i:0{len(str(num_iterations))}}"+'.csv')
                if not os.path.exists(chain_path):
                    os.makedirs(chain_path)
                self.chains.append(ChainFile(keys=keys, path=chain_path, mode=mode))

        self.samplers = []
        for i in range(self.num_chains):
            self.samplers.append(MetropolisHastings(
                self.models[i], theta0, dataloader, symmetric=True,
                kernel=DEMCKernel(self.sigma, c=0.1, dtype=self.models[i].dtype, device=self.models[i].device),
                chain=self.chains[i]
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
        self.samplers[i].kernel.set_density(self.samplers[i].current['theta'].clone().detach())

    def draw(self, savestate=False):
        for i in range(self.num_chains):
            self.set_kernel(i)
            self.samplers[i].draw(savestate=savestate)
            if savestate:
                self.chains[i].update(
                    {k: v.clone().detach() if isinstance(v, torch.Tensor) else v
                    for k, v in self.samplers[i].current.items()}
                )

    def run(self, num_iterations, num_burnin, verbose=False, verbose_step=100):
        verbose_msg = "Iteration {:" + str(len(str(num_iterations))) + "}, duration {}"

        for n in range(num_iterations):
            if verbose and (((n+1) % verbose_step) == 0):
                start_time = timer()

            savestate = False if (n < num_burnin) else True

            self.set_temperature(n, num_iterations)

            self.draw(savestate=savestate)

            if verbose and (((n+1) % verbose_step) == 0):
                end_time = timer()
                print(verbose_msg.format(n+1, timedelta(seconds=end_time-start_time)))
