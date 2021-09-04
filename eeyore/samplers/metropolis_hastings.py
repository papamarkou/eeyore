import torch

from .single_chain_serial_sampler import SingleChainSerialSampler
from eeyore.chains import ChainList
from eeyore.datasets import DataCounter
from eeyore.kernels import NormalKernel

class MetropolisHastings(SingleChainSerialSampler):
    def __init__(self, model,
        theta0=None, dataloader=None, data0=None, counter=None,
        symmetric=True, kernel=None, chain=ChainList()):
        super(MetropolisHastings, self).__init__(counter or DataCounter.from_dataloader(dataloader))
        self.model = model
        self.dataloader = dataloader
        self.symmetric = symmetric

        self.keys = ['sample', 'target_val', 'accepted']
        self.chain = chain

        if theta0 is not None:
            self.set_current(theta0.clone().detach(), data=data0)

        self.kernel = kernel or self.default_kernel(self.current)

    def default_kernel(self, state):
        loc = state['sample']
        scale = torch.ones(self.model.num_params(), dtype=self.model.dtype, device=self.model.device)
        return NormalKernel(loc, scale)

    def set_current(self, theta, data=None):
        x, y = super().set_current(theta, data=data)
        self.current['target_val'] = self.model.log_target(self.current['sample'].clone().detach(), x, y)

    def reset(self, theta, data=None, reset_counter=True, reset_chain=True):
        super().reset(theta, data=data, reset_counter=reset_counter, reset_chain=reset_chain)
        self.set_kernel(self.current)

    def set_kernel(self, state, scale=None, scale_tril=None):
        self.kernel.set_density_params(state['sample'].clone().detach())

    def draw(self, x, y, savestate=False):
        proposed = {key : None for key in self.keys}

        if self.counter.num_batches != 1:
            self.current['target_val'] = self.model.log_target(self.current['sample'].clone().detach(), x, y)

        proposed['sample'] = self.kernel.sample()
        proposed['target_val'] = self.model.log_target(proposed['sample'].clone().detach(), x, y)

        log_rate = proposed['target_val'] - self.current['target_val']
        if not self.symmetric:
            log_rate = log_rate - self.kernel.log_prob(proposed['sample'])
            self.set_kernel(proposed)
            log_rate = log_rate + self.kernel.log_prob(self.current['sample'])

        if torch.log(torch.rand(1, dtype=self.model.dtype, device=self.model.device)) < log_rate:
            self.current['sample'] = proposed['sample'].clone().detach()
            if self.counter.num_batches == 1:
                self.current['target_val'] = proposed['target_val'].clone().detach()
            if self.symmetric:
                self.set_kernel(proposed)
            self.current['accepted'] = 1
        else:
            self.model.set_params(self.current['sample'].clone().detach())
            if not self.symmetric:
                self.set_kernel(self.current)
            self.current['accepted'] = 0

        if savestate:
            self.chain.detach_and_update(self.current)

        self.current['sample'].detach_()
        self.current['target_val'].detach_()
