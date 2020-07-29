import copy
import itertools
import numpy as np
import torch

from pathlib import Path
from torch.distributions import Categorical

from .gamc import GAMC
from .mala import MALA
from .metropolis_hastings import MetropolisHastings
from .multi_chain_serial_sampler import MultiChainSerialSampler
from .smmala import SMMALA
from eeyore.chains import ChainFile, ChainList
from eeyore.datasets import DataCounter

class PowerPosteriorSampler(MultiChainSerialSampler):
    def __init__(self, model, dataloader, samplers,
        theta0=None, data0=None, counter=None,
        temperature=None, between_step=10, b=0.5, storage='list', keys=['sample', 'target_val', 'accepted'],
        path=Path.cwd(), mode='a', check_input=False):
        super(PowerPosteriorSampler, self).__init__(counter or DataCounter.from_dataloader(dataloader))
        self.between_step = between_step
        self.b = b

        self.num_chains = len(samplers)

        self.dataloader = dataloader

        self.sampler_names = [samplers[i][0] for i in range(self.num_chains)]

        self.use_gamc = all(name == 'GAMC' for name in self.sampler_names)
        if any(name == 'GAMC' for name in self.sampler_names) and (not self.use_gamc):
            raise ValueError

        self.init_samplers(model, samplers, theta0, data0 or next(iter(self.dataloader)), storage, keys, path, mode)

        self.set_temperature(temperature)

        self.dtype = self.samplers[0].samplers[0].model.dtype if self.use_gamc else self.samplers[0].model.dtype
        self.device = self.samplers[0].samplers[0].model.device if self.use_gamc else self.samplers[0].model.device

        if check_input:
            self.check_dtype()
            self.check_device()

        self.categoricals = []
        for i in range(self.num_chains):
            self.categoricals.append(Categorical(self.eval_categorical_probs(i)))

    def check_dtype(self):
        if self.use_gamc:
            valid = all(model.dtype == self.dtype for model in itertools.chain.from_iterable(
                [[sampler.samplers[i].model for i in range(2)] for sampler in self.samplers]
            ))
        else:
            valid = all(t == self.dtype for t in [sampler.model.dtype for sampler in self.samplers])

        if not valid:
            raise ValueError

    def check_device(self):
        if self.use_gamc:
            valid = all(model.device == self.device for model in itertools.chain.from_iterable(
                [[sampler.samplers[i].model for i in range(2)] for sampler in self.samplers]
            ))
        else:
            valid = all(t == self.device for t in [sampler.model.device for sampler in self.samplers])

        if not valid:
            raise ValueError

    def init_chain(self, i, storage, keys, path, mode):
        if storage == 'list':
            chain = ChainList(keys=keys)
        elif storage == 'file':
            chain_path = path.joinpath('chain'+f"{(i+1):0{len(str(self.num_chains))}}")
            if not chain_path.exists():
                chain_path.mkdir(parents=True, exist_ok=True)
            chain = ChainFile(keys=keys, path=chain_path, mode=mode)

        return chain

    def init_samplers(self, model, samplers, theta0, data0, storage, keys, path, mode):
        self.samplers = []
        for i in range(self.num_chains):
            if samplers[i][0] == 'MetropolisHastings':
                self.samplers.append(MetropolisHastings(
                    copy.deepcopy(model),
                    theta0=theta0, dataloader=None, data0=data0, counter=self.counter,
                    chain=self.init_chain(i, storage, keys, path, mode), **(samplers[i][1])
                ))
            elif samplers[i][0] == 'MALA':
                self.samplers.append(MALA(
                    copy.deepcopy(model),
                    theta0=theta0, dataloader=None, data0=data0, counter=self.counter,
                    chain=self.init_chain(i, storage, keys, path, mode), **(samplers[i][1])
                ))
            elif samplers[i][0] == 'SMMALA':
                self.samplers.append(SMMALA(
                    copy.deepcopy(model),
                    theta0=theta0, dataloader=None, data0=data0, counter=self.counter,
                    chain=self.init_chain(i, storage, keys, path, mode), **(samplers[i][1])
                ))
            elif samplers[i][0] == 'GAMC':
                self.samplers.append(GAMC(
                    copy.deepcopy(model),
                    theta0=theta0, dataloader=None, data0=data0, counter=self.counter,
                    chain=self.init_chain(i, storage, keys, path, mode), **(samplers[i][1])
                ))

    def default_indicator(self):
        return  self.num_chains - 1

    def set_temperature(self, temperature):
        if (temperature is not None) and (self.num_chains != len(temperature)):
            raise ValueError

        if (temperature is None):
            self.temperature = [(i/self.num_chains)**4 for i in range(1, self.num_chains+1)]
        else:
            self.temperature = temperature

        for i in range(self.num_chains):
            if self.use_gamc:
                for j in range(2):
                    self.samplers[i].samplers[j].model.temperature = self.temperature[i]
            else:
                self.samplers[i].model.temperature = self.temperature[i]

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
        return torch.tensor(
            [self.eval_categorical_prob(j, i) for j in itertools.chain(range(i), range(i+1, self.num_chains))]
        )

    def categorical_log_prob(self, j, i):
        return self.categoricals[i].log_prob(torch.tensor(self.from_events_to_seq(j, i)))

    def sample_categorical(self, i):
        return self.from_seq_to_events(self.categoricals[i].sample().item(), i)

    def reset(self, theta, data=None):
        super().reset(theta, data=data, reset_counter=False, reset_chain=True)
        self.counter.reset()

    def within_chain_move(self, i, x, y):
        if self.sampler_names[i] == 'GAMC':
            self.samplers[i].reset_in_sampler_from_data(x, y, reset_counter=False, reset_chain=False)
        self.samplers[i].draw(x, y, savestate=False)

    def within_chain_moves(self, x, y):
        for i in range(self.num_chains):
            self.within_chain_move(i, x, y)

    def between_chain_move_log_rate(self, i, j, sampler_i, sampler_j, x, y):
        return self.categorical_log_prob(i, j) - \
            self.categorical_log_prob(j, i) - \
            sampler_i.current['target_val'] - \
            sampler_j.current['target_val'] + \
            sampler_i.model.log_target(sampler_j.current['sample'].clone().detach(), x, y) + \
            sampler_j.model.log_target(sampler_i.current['sample'].clone().detach(), x, y)

    def swap_states(self, i, j, x, y):
        if self.use_gamc:
            state_copy = copy.deepcopy(
                self.samplers[i].samplers[self.samplers[i].current_kernel].current['sample'].clone().detach()
            )
            self.samplers[i].reset_in_sampler(
                self.samplers[j].samplers[self.samplers[j].current_kernel].current['sample'].clone().detach(),
                data=(x, y),
                sampler_id=self.samplers[j].current_kernel,
                reset_counter=False,
                reset_chain=False
            )
            self.samplers[j].reset_in_sampler(
                state_copy.clone().detach(),
                data=(x, y),
                sampler_id=self.samplers[i].current_kernel,
                reset_counter=False,
                reset_chain=False
            )

            indicator_copy = self.samplers[i].current_kernel
            self.samplers[i].current_kernel = self.samplers[j].current_kernel
            self.samplers[j].current_kernel = indicator_copy
        else:
            state_copy = copy.deepcopy(self.samplers[i].current['sample'].clone().detach())
            self.samplers[i].reset(
                self.samplers[j].current['sample'].clone().detach(),
                data=(x, y),
                reset_counter=False,
                reset_chain=False
            )
            self.samplers[j].reset(state_copy.clone().detach(), data=(x, y), reset_counter=False, reset_chain=False)

    def revert_states(self, sampler_i, sampler_j):
        sampler_i.model.set_params(sampler_i.current['sample'].clone().detach())
        sampler_j.model.set_params(sampler_j.current['sample'].clone().detach())

    def between_chain_move(self, i, j, x, y):
        if self.use_gamc:
            log_rate = self.between_chain_move_log_rate(
                i, j, self.samplers[i].current_sampler(), self.samplers[j].current_sampler(), x, y
            )
        else:
            log_rate = self.between_chain_move_log_rate(i, j, self.samplers[i], self.samplers[j], x, y)

        if torch.log(torch.rand(1, dtype=self.dtype, device=self.device)) < log_rate:
            self.swap_states(i, j, x, y)
        else:
            if self.use_gamc:
                self.revert_states(self.samplers[i].current_sampler(), self.samplers[j].current_sampler())
            else:
                self.revert_states(self.samplers[i], self.samplers[j])

    def between_chain_moves(self, x, y):
        for i in range(self.num_chains):
            j = self.sample_categorical(i)

            self.between_chain_move(i, j, x, y)

    def save_state(self, i):
        if self.use_gamc:
            self.samplers[i].chain.detach_and_update(self.samplers[i].current_sampler().current)
        else:
            self.samplers[i].chain.detach_and_update(self.samplers[i].current)

    def draw(self, x, y, savestate=False):
        self.within_chain_moves(x, y)

        if ((self.counter.idx % self.between_step) == 0):
            self.between_chain_moves(x, y)

        if savestate:
            for i in range(self.num_chains):
                self.save_state(i)
