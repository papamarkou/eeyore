import torch

from .single_chain_serial_sampler import SingleChainSerialSampler
from eeyore.chains import ChainList
from eeyore.datasets import DataCounter
from eeyore.stats import recursive_mean

class AM(SingleChainSerialSampler):
    def __init__(self, model,
        theta0=None, dataloader=None, data0=None, counter=None,
        cov0=None, l=0.05, b=1., c=1., t0=2, transform=None,
        chain=ChainList()):
        super(AM, self).__init__(counter or DataCounter.from_dataloader(dataloader))
        self.model = model
        self.dataloader = dataloader
        self.l = l
        self.b = b
        self.c = c
        self.t0 = t0
        self.transform = transform

        if cov0 is not None:
            self.cov0 = cov0.clone().detach()
        else:
            self.cov0 = torch.eye(self.model.num_params(), dtype=self.model.dtype, device=self.model.device)
        if self.transform is not None:
            self.cov0 = self.transform(self.cov0)
        self.num_accepted = 0
        self.keys = ['sample', 'target_val', 'accepted']
        self.chain = chain

        if theta0 is not None:
            self.set_all(theta0.clone().detach(), data=data0)

    def set_current(self, theta, data=None):
        x, y = super().set_current(theta, data=data)
        self.current['target_val'] = self.model.log_target(self.current['sample'].clone().detach(), x, y)

    def set_cov(self, cov=None):
        if cov is not None:
            self.cov = cov
        else:
            self.cov = self.cov0.clone().detach()
        self.running_mean = torch.zeros(self.model.num_params(), dtype=self.model.dtype, device=self.model.device)
        self.cov_sum = torch.zeros(
            self.model.num_params(), self.model.num_params(), dtype=self.model.dtype, device=self.model.device
        )

    def set_all(self, theta, data=None, cov=None):
        super().set_all(theta, data=data)
        self.set_cov(cov=cov)

    def reset(self, theta, data=None, reset_counter=True, reset_chain=True):
        super().reset(theta, data=data, reset_counter=reset_counter, reset_chain=reset_chain)
        self.num_accepted = 0

    def set_recursive_cov(self, n, offset=0):
        k = n - offset
        self.cov = (self.cov_sum - (k + 1) * torch.ger(self.running_mean, self.running_mean)) / k

    def draw(self, x, y, savestate=False, offset=0):
        proposed = {key : None for key in self.keys}

        randn_sample = torch.randn(self.model.num_params(), dtype=self.model.dtype, device=self.model.device)
        if (self.counter.idx + 1 - offset > self.t0):
            if torch.rand(1, dtype=self.model.dtype, device=self.model.device) < self.l:
                proposed['sample'] = self.current['sample'].clone().detach() + self.c * randn_sample
            else:
                proposed['sample'] = \
                    self.current['sample'].clone().detach() + self.b * torch.cholesky(self.cov) @ randn_sample
        else:
            proposed['sample'] = self.current['sample'].clone().detach() + self.c * randn_sample
        proposed['target_val'] = self.model.log_target(proposed['sample'].clone().detach(), x, y)

        log_rate = proposed['target_val'] - self.current['target_val']

        if torch.log(torch.rand(1, dtype=self.model.dtype, device=self.model.device)) < log_rate:
            self.current['sample'] = proposed['sample'].clone().detach()
            self.current['target_val'] = proposed['target_val'].clone().detach()
            self.current['accepted'] = 1
            if (self.counter.idx > 0):
                self.num_accepted = self.num_accepted + 1
        else:
            self.model.set_params(self.current['sample'].clone().detach())
            self.current['accepted'] = 0

        self.running_mean = recursive_mean(
            self.running_mean, self.counter.idx + 1, self.current['sample'], offset=offset
        )
        self.cov_sum = self.cov_sum + torch.ger(self.current['sample'], self.current['sample'])
        if (self.counter.idx + 1 - offset >= self.t0):
            if (self.num_accepted == 0):
                self.cov = self.cov0.clone().detach()
            else:
                self.set_recursive_cov(self.counter.idx, offset=offset)
                if self.transform is not None:
                    self.cov = self.transform(self.cov)

        if savestate:
            self.chain.detach_and_update(self.current)

        self.current['sample'].detach_()
        self.current['target_val'].detach_()
