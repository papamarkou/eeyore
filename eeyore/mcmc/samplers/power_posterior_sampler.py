import torch

from eeyore.api import Sampler
from eeyore.mcmc import MCChain
from .metropolis_hastings import MetropolisHastings
from .mala import MALA

class PowerPosteriorSampler(Sampler):
    def __init__(self, model, theta0, dataloader, samplers, temperatures, keys=['theta', 'target_val', 'accepted']):
        super(PowerPosteriorSampler, self).__init__()
        self.temperatures = temperatures

        self.num_chains = len(self.temperatures)

        if (len(samplers) != self.num_chains):
            raise ValueError

        self.models = self.num_chains*[model]
        for i in range(self.num_chains):
            self.models[i].temperatures = temperatures[i]

        self.samplers = []
        for i in range(self.num_chains):
            if samplers[i][0] == 'MetropolisHastings':
                self.samplers.append(MetropolisHastings(self.models[i], theta0, dataloader, **(samplers[i][1])))
            elif samplers[i][0] == 'MALA':
                self.samplers.append(MALA(self.models[i], theta0, dataloader, **(samplers[i][1])))
            else:
                ValueError
