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
    def __init__(self, model, theta0, dataloader, samplers, data0=None, counter=None, choose_kernel=None, a=10.,
        chain=ChainList(keys=['sample', 'target_val', 'accepted'])):
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

        self.init_samplers(samplers, theta0, data0 or next(iter(self.dataloader)), model)

    def init_samplers(self, samplers, theta0, data0, model):
        self.samplers = []
        for i in range(2):
            if self.sampler_names[i] == 'MetropolisHastings':
                self.samplers.append(MetropolisHastings(
                    copy.deepcopy(model),
                    theta0,
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
            if self.sampler_names[i] == 'AM':
                self.samplers.append(AM(
                    copy.deepcopy(model), theta0,
                    dataloader=None, data0=data0, counter=self.counter,
                    chain=self.chain, **(samplers[i][1])
                ))
            if self.sampler_names[i] == 'RAM':
                self.samplers.append(RAM(
                    copy.deepcopy(model), theta0,
                    dataloader=None, data0=data0, counter=self.counter,
                    chain=self.chain, **(samplers[i][1])
                ))
            elif self.sampler_names[i] == 'MALA':
                self.samplers.append(MALA(
                    copy.deepcopy(model), theta0,
                    dataloader=None, data0=data0, counter=self.counter,
                    chain=self.chain, **(samplers[i][1])
                ))
            elif self.sampler_names[i] == 'SMMALA':
                self.samplers.append(SMMALA(
                    copy.deepcopy(model), theta0,
                    dataloader=None, data0=data0, counter=self.counter,
                    chain=self.chain, **(samplers[i][1])
                ))

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

    def set_current(self, theta, data=None, sampler_id=None):
        x, y = super().set_current(theta, data=data)

        i = sampler_id if sampler_id is not None else self.current_kernel

        if self.sampler_names[i] == 'AM':
            self.samplers[i].set_current(theta, data=(x, y), cov=self.samplers[i].cov0.clone().detach())
        elif self.sampler_names[i] == 'RAM':
            self.samplers[i].set_current(theta, data=(x, y), cov=self.samplers[i].cov0)
        else:
            self.samplers[i].set_current(theta, data=(x, y))

    def set_current_from_data(self, x, y):
        self.set_kernel_indicators(self.counter.idx+1, self.counter.num_iters)

        if (self.counter.idx > 0) and (self.current_kernel != self.last_kernel):
            self.set_current(
                self.samplers[self.last_kernel].current['sample'].clone().detach(),
                data=(x, y),
                sampler_id=self.current_kernel
            )

    def reset(self, theta, data=None, sampler_id=None):
        self.set_current(theta, data=data, sampler_id=sampler_id)
        super().reset()

    def draw(self, x, y, savestate=False):
        self.set_current_from_data(x, y)

        if ((self.sampler_names[self.current_kernel] == 'AM') or
            (self.sampler_names[self.current_kernel] == 'RAM')
        ):
            self.current_sampler().draw(x, y, savestate=savestate, offset=self.offset)
        else:
            self.current_sampler().draw(x, y, savestate=savestate)
