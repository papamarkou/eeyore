import os

import torch

from eeyore.api import Sampler
from eeyore.kernels import DEMCKernel
from eeyore.mcmc import ChainFile, ChainList
from .metropolis_hastings import MetropolisHastings

class DEMC(Sampler):
    def __init__(self, model, theta0, dataloader, num_chains=10, c=0.1, schedule=lambda n, num_iterations: 1.,
    storage='list', keys=['theta', 'target_val', 'accepted'], path=os.getcwd(), mode='a'):
        super(DEMC, self).__init__()
        self.num_chains = num_chains
        self.c = c
        self.schedule = schedule

        self.models = []
        for i in range(self.num_chains):
            self.models.append(copy.deepcopy(model))

        self.samplers = []
        for i in range(self.num_chains):
            self.samplers.append(MetropolisHastings(self.models[i], theta0, dataloader))

    def draw(self, savestate=False):
        self.within_chain_moves()

        if savestate:
            for i in range(self.num_chains):
                self.chains[i].update(
                    {k: v.clone().detach() if isinstance(v, torch.Tensor) else v
                    for k, v in self.sampler.current.items()}
                )

    def run(self, num_iterations, num_burnin, verbose=False, verbose_step=100):
        verbose_msg = "Iteration {:" + str(len(str(num_iterations))) + "}, duration {}"

        for n in range(num_iterations):
            if verbose and (((n+1) % verbose_step) == 0):
                start_time = timer()

            savestate = False if (n < num_burnin) else True

            self.draw(savestate=savestate)

            if verbose and (((n+1) % verbose_step) == 0):
                end_time = timer()
                print(verbose_msg.format(n+1, timedelta(seconds=end_time-start_time)))
