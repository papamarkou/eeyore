import torch

from .single_chain_serial_sampler import SingleChainSerialSampler
from eeyore.chains import ChainList
from eeyore.datasets import DataCounter

class RAM(SingleChainSerialSampler):
    def __init__(self, model,
        theta0=None, dataloader=None, data0=None, counter=None,
        cov0=None, a=0.234, g=0.7, chain=ChainList()):
        super(RAM, self).__init__(counter or DataCounter.from_dataloader(dataloader))
        self.model = model
        self.dataloader = dataloader
        self.a = a
        self.g = g

        if cov0 is not None:
            self.cov0 = cov0.clone().detach()
        else:
            self.cov0 = torch.eye(self.model.num_params(), dtype=self.model.dtype, device=self.model.device)
        self.keys = ['sample', 'target_val', 'accepted']
        self.chain = chain

        if theta0 is not None:
            self.set_all(theta0.clone().detach(), data=data0)

    def set_current(self, theta, data=None):
        x, y = super().set_current(theta, data=data)
        self.current['target_val'] = self.model.log_target(self.current['sample'].clone().detach(), x, y)

    def set_cov(self, cov=None):
        self.chol_cov = torch.cholesky(cov or self.cov0)

    def set_all(self, theta, data=None, cov=None):
        super().set_all(theta, data=data)
        self.set_cov(cov=cov)

    def draw(self, x, y, savestate=False, offset=0):
        proposed = {key : None for key in self.keys}

        randn_sample = torch.randn(self.model.num_params(), dtype=self.model.dtype, device=self.model.device)
        proposed['sample'] = self.current['sample'].clone().detach() + self.chol_cov @ randn_sample
        proposed['target_val'] = self.model.log_target(proposed['sample'].clone().detach(), x, y)

        log_rate = proposed['target_val'] - self.current['target_val']

        if torch.log(torch.rand(1, dtype=self.model.dtype, device=self.model.device)) < log_rate:
            self.current['sample'] = proposed['sample'].clone().detach()
            self.current['target_val'] = proposed['target_val'].clone().detach()
            self.current['accepted'] = 1
        else:
            self.model.set_params(self.current['sample'].clone().detach())
            self.current['accepted'] = 0

        h = min(1, self.model.num_params() * (self.counter.idx + 1 - offset) ** (-self.g))
        self.chol_cov = torch.cholesky(self.chol_cov @ (
            torch.eye(self.model.num_params(), dtype=self.model.dtype, device=self.model.device) + \
            h * (min(1, torch.exp(log_rate).item()) - self.a
            ) * torch.ger(randn_sample, randn_sample) / \
            torch.dot(randn_sample, randn_sample).item()) @ self.chol_cov.t())

        if savestate:
            self.chain.detach_and_update(self.current)

        self.current['sample'].detach_()
        self.current['target_val'].detach_()
