import torch

from eeyore.api import Sampler
from eeyore.mcmc import MCChain
from .metropolis_hastings import MetropolisHastings
from .mala import MALA

class PowerPosteriorSampler(Sampler):
    def __init__(self, model, theta0, dataloader, samplers, temperatures=None):
        super(PowerPosteriorSampler, self).__init__()

        self.num_chains = len(samplers)

        if (temperatures is not None) and (self.num_chains != len(temperatures)):
            raise ValueError

        if (temperatures is None):
            self.temperatures = [(i/(self.num_chains-1))**4 for i in range(self.num_chains)]
        else:
            self.temperatures = temperatures

        self.models = self.num_chains*[model]
        for i in range(self.num_chains):
            self.models[i].temperatures = self.temperatures[i]

        self.samplers = []
        for i in range(self.num_chains):
            if samplers[i][0] == 'MetropolisHastings':
                self.samplers.append(MetropolisHastings(self.models[i], theta0, dataloader, **(samplers[i][1])))
            elif samplers[i][0] == 'MALA':
                self.samplers.append(MALA(self.models[i], theta0, dataloader, **(samplers[i][1])))
            else:
                ValueError

        self.chains = []
        for i in range(self.num_chains):
            self.chains.append(MCChain(self.samplers[i].keys))

    def reset(self, theta):
        for sampler in self.samplers:
            sampler.reset(theta)
