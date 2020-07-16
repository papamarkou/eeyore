import numpy as np
import torch

from scipy.stats import truncnorm

from .single_chain_serial_sampler import SingleChainSerialSampler
from eeyore.chains import ChainList
from eeyore.datasets import DataCounter

class MALA(SingleChainSerialSampler):
    def __init__(self, model,
        theta0=None, dataloader=None, data0=None, counter=None,
        step=0.1, chain=ChainList()):
        super().__init__(counter or DataCounter.from_dataloader(dataloader))
        self.model = model
        self.dataloader = dataloader
        self.step = step

        self.keys = ['sample', 'target_val', 'grad_val', 'accepted']
        self.chain = chain

        if theta0 is not None:
            self.set_current(theta0.clone().detach(), data=data0)

    def set_current(self, theta, data=None):
        x, y = super().set_current(theta, data=data)
        self.current['target_val'], self.current['grad_val'] = \
            self.model.upto_grad_log_target(self.current['sample'].clone().detach(), x, y)

    def draw(self, x, y, savestate=False):
        proposed = {key : None for key in self.keys}

        proposal_mean = self.current['sample'] + 0.5 * self.step * self.current['grad_val']

        if (self.model.constraint is None) or (self.model.constraint == 'transformation'):
            proposed['sample'] = \
                proposal_mean + np.sqrt(self.step) * \
                torch.randn(self.model.num_params(), dtype=self.model.dtype, device=self.model.device)
        elif self.model.constraint == 'truncation':
            l = -np.inf if (self.model.bounds[0] is None) else self.model.bounds[0]
            u = np.inf if (self.model.bounds[1] is None) else self.model.bounds[1]

            loc = proposal_mean.detach().cpu().numpy()
            scale = np.sqrt(self.step)
            a = (l - loc) / scale
            b = (u - loc) / scale
            proposed['sample'] = \
                torch.from_numpy(truncnorm.rvs(a=a, b=b, loc=loc, scale=scale, size=self.model.num_params()) \
                ).to(dtype=self.model.dtype, device=self.model.device)

        proposed['target_val'], proposed['grad_val'] = \
            self.model.upto_grad_log_target(proposed['sample'].clone().detach(), x, y)

        log_rate = proposed['target_val'] - self.current['target_val']
        if (self.model.constraint is None) or (self.model.constraint == 'transformation'):
            log_rate = log_rate + 0.5 * torch.sum((proposed['sample'] - proposal_mean) ** 2) / self.step
        elif self.model.constraint == 'truncation':
            log_rate = log_rate - torch.sum(torch.from_numpy( \
            truncnorm.logpdf(proposed['sample'].detach().cpu().numpy(), a=a, b=b, loc=loc, scale=scale) \
            ).to(dtype=self.model.dtype, device=self.model.device))

        proposal_mean = proposed['sample'] + 0.5 * self.step * proposed['grad_val']

        if (self.model.constraint is None) or (self.model.constraint == 'transformation'):
            log_rate = log_rate - 0.5 * torch.sum((self.current['sample'] - proposal_mean) ** 2) / self.step
        elif self.model.constraint == 'truncation':
            loc = proposal_mean.detach().cpu().numpy()
            a = (l - loc) / scale
            b = (u - loc) / scale
            log_rate = log_rate + torch.sum(torch.from_numpy( \
            truncnorm.logpdf(self.current['sample'].detach().cpu().numpy(), a=a, b=b, loc=loc, scale=scale) \
            ).to(dtype=self.model.dtype, device=self.model.device))

        if torch.log(torch.rand(1, dtype=self.model.dtype, device=self.model.device)) < log_rate:
            self.current['sample'] = proposed['sample'].clone().detach()
            self.current['target_val'] = proposed['target_val'].clone().detach()
            self.current['grad_val'] = proposed['grad_val'].clone().detach()
            self.current['accepted'] = 1
        else:
            self.model.set_params(self.current['sample'].clone().detach())
            self.current['accepted'] = 0

        if savestate:
            self.chain.detach_and_update(self.current)

        self.current['sample'].detach_()
        self.current['target_val'].detach_()
        self.current['grad_val'].detach_()
