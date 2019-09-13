from itertools import chain

import numpy as np

from timeit import default_timer as timer
from datetime import timedelta

import torch
from torch.distributions import Categorical

from eeyore.api import Sampler
from eeyore.mcmc import MCChain
from .metropolis_hastings import MetropolisHastings
from .mala import MALA

class PowerPosteriorSampler(Sampler):
    def __init__(self, model, theta0, dataloader, samplers, temperatures=None, b=0.5):
        super(PowerPosteriorSampler, self).__init__()
        self.dataloader = dataloader
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
            self.categoricals.append(Categorical(self.eval_categorical_probs(i)))

        self.chains = []
        for i in range(self.num_chains):
            self.chains.append(MCChain(self.samplers[i].keys))

    def from_seq_to_events(self, k, i):
        j = k if (k < i) else (k+1)
        return j

    def from_events_to_seq(self, j, i):
        k = j if (j < i) else (j-1)
        return k

    def eval_categorical_prob(self, j, i):
        eb = np.exp(-self.b)
        numerator = eb**np.absolute(j-i)
        denominator = eb*(2-eb**i-eb**(self.num_chains-1-i))/(1-eb)
        return numerator/denominator

    def eval_categorical_probs(self, i):
        return torch.tensor([self.eval_categorical_prob(j, i) for j in chain(range(i), range(i+1, self.num_chains))])

    def categorical_log_prob(self, j, i):
        return self.categoricals[i].log_prob(torch.tensor(self.from_events_to_seq(j, i)))

    def sample_categorical(self, i):
        return self.from_seq_to_events(self.categoricals[i].sample().item(), i)

    def reset(self, theta):
        for sampler in self.samplers:
            sampler.reset(theta)

    def within_chain_moves(self):
        for sampler in self.samplers:
            sampler.draw(savestate=False)

    def between_chain_move(self, i, j):
        data, label = next(iter(self.dataloader))

        log_rate = self.categorical_log_prob(i, j) - \
            self.categorical_log_prob(j, i) - \
            self.samplers[i].current['target_val'] - \
            self.samplers[j].current['target_val'] + \
            self.samplers[i].model.log_target(self.samplers[j].current['theta'].clone().detach(), data, label) + \
            self.samplers[j].model.log_target(self.samplers[i].current['theta'].clone().detach(), data, label)

        if torch.log(
            torch.rand(1, dtype=self.samplers[0].model.dtype, device=self.samplers[0].model.device)
            ) < log_rate:
            theta = self.samplers[i].current['theta'].clone().detach()
            self.samplers[i].reset(self.samplers[j].current['theta'])
            self.samplers[j].reset(theta)
        else:
            self.samplers[i].model.set_params(self.samplers[i].current['theta'].clone().detach())
            self.samplers[j].model.set_params(self.samplers[j].current['theta'].clone().detach())

    def between_chain_moves(self):
        for i in range(self.num_chains):
            j = self.sample_categorical(i)

            self.between_chain_move(i, j)

    def draw(self, between=True, savestate=False):
        self.within_chain_moves()

        if between:
            self.between_chain_moves()

        if savestate:
            for i in range(self.num_chains):
                self.chains[i].update(
                    {k: v.clone().detach() if isinstance(v, torch.Tensor) else v
                    for k, v in self.samplers[i].current.items()}
                )

    def run(self, num_iterations, num_burnin, between_step=10, verbose=False):
        verbose_msg = "Iteration {:" + str(len(str(num_iterations))) + "}, duration {}"

        for n in range(num_iterations):
            if verbose:
                start_time = timer()

            between = True if ((n % between_step) == 0) else False
            savestate = False if (n < num_burnin) else True

            self.draw(between=between, savestate=savestate)

            if verbose:
                end_time = timer()
                print(verbose_msg.format(n+1, timedelta(seconds=end_time-start_time)))
