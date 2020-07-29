import copy
import torch

from torch.distributions import Bernoulli

from .am import AM
from .mala import MALA
from .metropolis_hastings import MetropolisHastings
from .ram import RAM
from .single_chain_serial_sampler import SingleChainSerialSampler
from .smmala import SMMALA
from eeyore.chains import ChainList
from eeyore.datasets import DataCounter
from eeyore.kernels import MultivariateNormalKernel

class GAMC(SingleChainSerialSampler):
    def __init__(self, model, samplers, theta0=None, dataloader=None, data0=None, counter=None, choose_kernel=None,
        a=10., chain=ChainList()):
        super(GAMC, self).__init__(counter or DataCounter.from_dataloader(dataloader))

        self.dataloader = dataloader

        if choose_kernel is not None:
            self.choose_kernel = choose_kernel
        else:
            self.choose_kernel = lambda n, num_iters : self.sample_exp_decay_indicator(n, num_iters, a)
        self.last_kernel = None
        self.current_kernel = None
        self.offset = 0

        self.keys = ['sample', 'target_val', 'accepted']
        self.chain = chain

        self.sampler_names = [samplers[i][0] for i in range(2)]

        self.init_samplers(model, samplers, theta0, data0 or next(iter(self.dataloader)))

    def init_samplers(self, model, samplers, theta0, data0):
        self.samplers = []
        for i in range(2):
            if samplers[i][0] == 'MetropolisHastings':
                self.samplers.append(MetropolisHastings(
                    copy.deepcopy(model),
                    theta0=theta0,
                    dataloader=None,
                    data0=data0,
                    counter=self.counter,
                    symmetric=True,
                    kernel=MultivariateNormalKernel(
                        torch.zeros(model.num_params(), dtype=model.dtype, device=model.device),
                        scale_tril=torch.eye(model.num_params(), dtype=model.dtype, device=model.device)
                    ),
                    chain=self.chain
                ))
            if samplers[i][0] == 'AM':
                self.samplers.append(AM(
                    copy.deepcopy(model),
                    theta0=theta0, dataloader=None, data0=data0, counter=self.counter,
                    chain=self.chain, **(samplers[i][1])
                ))
            if samplers[i][0] == 'RAM':
                self.samplers.append(RAM(
                    copy.deepcopy(model),
                    theta0=theta0, dataloader=None, data0=data0, counter=self.counter,
                    chain=self.chain, **(samplers[i][1])
                ))
            elif samplers[i][0] == 'MALA':
                self.samplers.append(MALA(
                    copy.deepcopy(model),
                    theta0=theta0, dataloader=None, data0=data0, counter=self.counter,
                    chain=self.chain, **(samplers[i][1])
                ))
            elif samplers[i][0] == 'SMMALA':
                self.samplers.append(SMMALA(
                    copy.deepcopy(model),
                    theta0=theta0, dataloader=None, data0=data0, counter=self.counter,
                    chain=self.chain, **(samplers[i][1])
                ))

    def default_indicator(self):
        return self.current_kernel

    def get_model(self, idx=None):
        return self.samplers[idx or self.default_indicator()].model

    def current_sampler(self):
        return self.samplers[self.current_kernel]

    def exp_decay_indicator_prob(self, n, num_iters, a=10.):
        return torch.exp(torch.tensor([- a * n / num_iters]))

    def sample_exp_decay_indicator(self, n, num_iters, a=10.):
        return Bernoulli(self.exp_decay_indicator_prob(n, num_iters, a=a)).sample().to(dtype=torch.int)

    def set_kernel_indicators(self, n, num_iters):
        if self.current_kernel is not None:
            self.last_kernel = self.current_kernel

        self.current_kernel = self.choose_kernel(n, num_iters)

        if self.current_kernel == 1:
            self.offset = n - 1

    def set_all(self, theta, data=None):
        x, y = data or next(iter(self.dataloader))
        for sampler in self.samplers:
            sampler.set_all(theta, data=(x, y))

    def reset_in_sampler(self, theta, data=None, sampler_id=None, reset_counter=True, reset_chain=True):
        self.samplers[sampler_id or self.current_kernel].reset(
            theta, data=data or next(iter(self.dataloader)), reset_counter=reset_counter, reset_chain=reset_chain
        )

    def reset_in_sampler_from_data(self, x, y, reset_counter=True, reset_chain=True):
        self.set_kernel_indicators(self.counter.idx+1, self.counter.num_iters)

        if (self.counter.idx > 0) and (self.current_kernel != self.last_kernel):
            self.reset_in_sampler(
                self.samplers[self.last_kernel].current['sample'].clone().detach(),
                data=(x, y),
                reset_counter=reset_counter,
                reset_chain=reset_chain
            )

    def reset(self, theta, data=None, reset_counter=True, reset_chain=True):
        super().reset(theta, data=data, reset_counter=reset_counter, reset_chain=reset_chain)
        self.last_kernel = None
        self.current_kernel = None
        self.offset = 0

    def draw(self, x, y, savestate=False):
        self.reset_in_sampler_from_data(x, y, reset_counter=False, reset_chain=False)

        if ((self.sampler_names[self.current_kernel] == 'AM') or
            (self.sampler_names[self.current_kernel] == 'RAM')
        ):
            self.current_sampler().draw(x, y, savestate=savestate, offset=self.offset)
        else:
            self.current_sampler().draw(x, y, savestate=savestate)
