import copy

from itertools import chain

import numpy as np

from timeit import default_timer as timer
from datetime import timedelta

import torch
from torch.distributions import Categorical

from eeyore.api import Sampler
from eeyore.mcmc import ChainFile, ChainList
from .metropolis_hastings import MetropolisHastings
from .mala import MALA
from .smmala import SMMALA

class PowerPosteriorSampler(Sampler):
    def __init__(self, models, theta0s, dataloaders, samplers, temperatures=None, b=0.5):
        super(PowerPosteriorSampler, self).__init__()
        self.models = models
        self.dataloaders = dataloaders
        self.b = b

        self.num_powers = len(samplers)

        if (temperatures is not None) and (self.num_powers != len(temperatures)):
            raise ValueError

        if (temperatures is None):
            self.temperatures = [(i/self.num_powers)**4 for i in range(1, self.num_powers+1)]
        else:
            self.temperatures = temperatures

        for i in range(self.num_powers):
            self.models[i].temperature = self.temperatures[i]

        self.samplers = []
        for i in range(self.num_powers):
            if samplers[i][0] == 'MetropolisHastings':
                self.samplers.append(
                    MetropolisHastings(self.models[i], theta0s[i], self.dataloaders[i], **(samplers[i][1]))
                )
            elif samplers[i][0] == 'MALA':
                self.samplers.append(MALA(self.models[i], theta0s[i], self.dataloaders[i], **(samplers[i][1])))
            elif samplers[i][0] == 'SMMALA':
                self.samplers.append(SMMALA(self.models[i], theta0s[i], self.dataloaders[i], **(samplers[i][1])))
            else:
                ValueError

        self.categoricals = []
        for i in range(self.num_powers):
            self.categoricals.append(Categorical(self.eval_categorical_probs(i)))

        self.chains = []
        for i in range(self.num_powers):
            self.chains.append(ChainList(self.samplers[i].keys))

    def from_seq_to_events(self, k, i):
        j = k if (k < i) else (k+1)
        return j

    def from_events_to_seq(self, j, i):
        k = j if (j < i) else (j-1)
        return k

    def eval_categorical_prob(self, j, i):
        eb = np.exp(-self.b)
        numerator = eb**np.absolute(j-i)
        denominator = eb*(2-eb**i-eb**(self.num_powers-1-i))/(1-eb)
        return numerator/denominator

    def eval_categorical_probs(self, i):
        return torch.tensor([self.eval_categorical_prob(j, i) for j in chain(range(i), range(i+1, self.num_powers))])

    def categorical_log_prob(self, j, i):
        return self.categoricals[i].log_prob(torch.tensor(self.from_events_to_seq(j, i)))

    def sample_categorical(self, i):
        return self.from_seq_to_events(self.categoricals[i].sample().item(), i)

    def reset(self, theta):
        for sampler in self.samplers:
            sampler.reset(theta)

    def get_chain(self):
        return self.chains[self.num_powers-1]

    def within_chain_moves(self):
        for sampler in self.samplers:
            sampler.draw(savestate=False)

    def between_chain_move(self, i, j):
        log_rate = self.categorical_log_prob(i, j) - \
            self.categorical_log_prob(j, i) - \
            self.samplers[i].current['target_val'] - \
            self.samplers[j].current['target_val'] + \
            self.samplers[i].model.log_target(
                self.samplers[j].current['theta'].clone().detach(), self.samplers[i].dataloader
            ) + \
            self.samplers[j].model.log_target(
                self.samplers[i].current['theta'].clone().detach(), self.samplers[j].dataloader
            )

        if torch.log(
            torch.rand(1, dtype=self.samplers[i].model.dtype, device=self.samplers[i].model.device)
            ) < log_rate:
            state_copy = copy.deepcopy(self.samplers[i].current['theta'].clone().detach())
            self.samplers[i].reset(self.samplers[j].current['theta'].clone().detach())
            self.samplers[j].reset(state_copy.clone().detach())
        else:
            self.samplers[i].model.set_params(self.samplers[i].current['theta'].clone().detach())
            self.samplers[j].model.set_params(self.samplers[j].current['theta'].clone().detach())

    def between_chain_moves(self):
        for i in range(self.num_powers):
            j = self.sample_categorical(i)

            self.between_chain_move(i, j)

    def draw(self, between=True, savestate=False):
        self.within_chain_moves()

        if between:
            self.between_chain_moves()

        if savestate:
            for i in range(self.num_powers):
                self.chains[i].update(
                    {k: v.clone().detach() if isinstance(v, torch.Tensor) else v
                    for k, v in self.samplers[i].current.items()}
                )

    def run(self, num_iterations, num_burnin, between_step=10, verbose=False, verbose_step=100):
        verbose_msg = "Iteration {:" + str(len(str(num_iterations))) + "}, duration {}"

        for n in range(num_iterations):
            if verbose and (((n+1) % verbose_step) == 0):
                start_time = timer()

            between = True if ((n % between_step) == 0) else False
            savestate = False if (n < num_burnin) else True

            self.draw(between=between, savestate=savestate)

            if verbose and (((n+1) % verbose_step) == 0):
                end_time = timer()
                print(verbose_msg.format(n+1, timedelta(seconds=end_time-start_time)))
