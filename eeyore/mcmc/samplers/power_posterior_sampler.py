from itertools import chain

import numpy as np

import torch
from torch.distributions import Categorical

from eeyore.api import SerialSampler
from eeyore.mcmc import MCChain
from .metropolis_hastings import MetropolisHastings
from .mala import MALA

class PowerPosteriorSampler(SerialSampler):
    def __init__(self, model, theta0, dataloader, samplers, temperatures=None, b=0.5):
        super(PowerPosteriorSampler, self).__init__()
        self.b = b

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

        self.categoricals = []
        for i in range(self.num_chains):
            self.categoricals.append(Categorical(self.categorical_probs(i)))

        self.chains = []
        for i in range(self.num_chains):
            self.chains.append(MCChain(self.samplers[i].keys))

    def from_seq_to_events(self, k, i):
        j = k if (k < i) else (k+1)
        return j

    def from_events_to_seq(self, j, i):
        k = j if (j < i) else (j-1)
        return k

    def categorical_prob(self, j, i):
        eb = np.exp(-self.b)
        numerator = eb**np.absolute(j-i)
        denominator = eb*(2-eb**i-eb**(self.num_chains-1-i))/(1-eb)
        return numerator/denominator

    def categorical_probs(self, i):
        return torch.tensor([self.categorical_prob(j, i) for j in chain(range(i), range(i+1, self.num_chains))])

    def categorical_sample(self, i):
        return self.from_seq_to_events(self.categoricals[i].sample().item(), i)

    def reset(self, theta):
        for sampler in self.samplers:
            sampler.reset(theta)

    def draw(self, savestate=False):
        for sampler in self.samplers:
            sampler.draw(savestate=savestate)

        for i in range(self.num_chains):
            j = self.categorical_sample(i)

        log_rate = log_rate - self.samplers[i].current['target_val'] - self.samplers[j].current['target_val']
