import torch

from .single_chain_serial_sampler import SingleChainSerialSampler
from eeyore.chains import ChainList
from eeyore.datasets import DataCounter
from eeyore.stats import recursive_mean

class AM(SingleChainSerialSampler):
    def __init__(self, model, theta0,
        dataloader=None, data0=None, counter=None,
        cov0=None, l=0.05, b=1., c=1., t0=2, transform=None,
        chain=ChainList(keys=['sample', 'target_val', 'accepted'])):
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
        self.current = {key : None for key in self.keys}
        self.chain = chain

        x, y = data0 or next(iter(self.dataloader))
        self.reset(theta0.clone().detach(), x, y, cov=self.cov0.clone().detach())

    def set_current(self, theta, data=None):
        x, y = super().set_current(theta, data=data)
        self.current['target_val'], self.current['grad_val'] = \
            self.model.upto_grad_log_target(self.current['sample'].clone().detach(), x, y)

    def reset(self, theta, data=None):
        self.set_current(theta, data=data)
        super().reset()

    def reset(self, theta, x, y, cov=None, reset_chain=False):
        if reset_chain:
            super().reset()

        self.current['sample'] = theta
        self.current['target_val'] = self.model.log_target(self.current['sample'].clone().detach(), x, y)
        self.running_mean = torch.zeros(self.model.num_params(), dtype=self.model.dtype, device=self.model.device)
        self.cov_sum = torch.zeros(
            self.model.num_params(), self.model.num_params(), dtype=self.model.dtype, device=self.model.device
        )
        if cov is not None:
            self.cov = cov

    def set_cov(self, n, offset=0):
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
                self.set_cov(self.counter.idx, offset=offset)
                if self.transform is not None:
                    self.cov = self.transform(self.cov)

        if savestate:
            self.chain.detach_and_update(self.current)

        self.current['sample'].detach_()
        self.current['target_val'].detach_()
