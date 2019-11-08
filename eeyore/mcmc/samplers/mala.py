import numpy as np
from scipy.stats import truncnorm

import torch

from eeyore.api import SerialSampler
from eeyore.mcmc import ChainFile, ChainList

class MALA(SerialSampler):
    def __init__(self, model, theta0, dataloader, step=0.1, chain=ChainList(keys=['theta', 'target_val', 'accepted'])):
        super(MALA, self).__init__()
        self.model = model
        self.dataloader = dataloader
        self.step = step

        self.keys = ['theta', 'target_val', 'grad_val', 'accepted']
        self.current = {key : None for key in self.keys}
        self.chain = chain

        self.reset(theta0)

    def reset(self, theta):
        data, label = next(iter(self.dataloader))

        self.current['theta'] = theta.clone().detach()
        self.current['target_val'], self.current['grad_val'] = \
            self.model.upto_grad_log_target(self.current['theta'].clone().detach(), data, label)

    def draw(self, savestate=False):
        proposed = {key : None for key in self.keys}

        for data, label in self.dataloader:
            proposal_mean = self.current['theta'] + 0.5 * self.step * self.current['grad_val']

            if (self.model.constraint is None) or (self.model.constraint == 'transformation'):
                proposed['theta'] = \
                    proposal_mean + np.sqrt(self.step) * \
                    torch.randn(self.model.num_params(), dtype=self.model.dtype, device=self.model.device)
            elif self.model.constraint == 'truncation':
                l = -np.inf if (self.model.bounds[0] is None) else self.model.bounds[0]
                u = np.inf if (self.model.bounds[1] is None) else self.model.bounds[1]

                loc = proposal_mean.detach().cpu().numpy()
                scale = np.sqrt(self.step)
                a = (l - loc) / scale
                b = (u - loc) / scale
                proposed['theta'] = \
                    torch.from_numpy(truncnorm.rvs(a=a, b=b, loc=loc, scale=scale, size=self.model.num_params()) \
                    ).to(dtype=self.model.dtype).to(device=self.model.device)

            proposed['target_val'], proposed['grad_val'] = \
                self.model.upto_grad_log_target(proposed['theta'].clone().detach(), data, label)

            log_rate = proposed['target_val'] - self.current['target_val']
            if (self.model.constraint is None) or (self.model.constraint == 'transformation'):
                log_rate = log_rate + 0.5 * torch.sum((proposed['theta'] - proposal_mean) ** 2) / self.step
            elif self.model.constraint == 'truncation':
                log_rate = log_rate - torch.sum(torch.from_numpy( \
                truncnorm.logpdf(proposed['theta'].detach().cpu().numpy(), a=a, b=b, loc=loc, scale=scale) \
                ).to(dtype=self.model.dtype).to(device=self.model.device))

            proposal_mean = proposed['theta'] + 0.5 * self.step * proposed['grad_val']

            if (self.model.constraint is None) or (self.model.constraint == 'transformation'):
                log_rate = log_rate - 0.5 * torch.sum((self.current['theta'] - proposal_mean) ** 2) / self.step
            elif self.model.constraint == 'truncation':
                loc = proposal_mean.detach().cpu().numpy()
                a = (l - loc) / scale
                b = (u - loc) / scale
                log_rate = log_rate + torch.sum(torch.from_numpy( \
                truncnorm.logpdf(self.current['theta'].detach().cpu().numpy(), a=a, b=b, loc=loc, scale=scale) \
                ).to(dtype=self.model.dtype).to(device=self.model.device))

            if torch.log(torch.rand(1, dtype=self.model.dtype, device=self.model.device)) < log_rate:
                self.current['theta'] = proposed['theta'].clone().detach()
                self.current['target_val'] = proposed['target_val'].clone().detach()
                self.current['grad_val'] = proposed['grad_val'].clone().detach()
                self.current['accepted'] = 1
            else:
                self.model.set_params(self.current['theta'].clone().detach())
                self.current['accepted'] = 0

            if savestate:
                self.chain.update(
                    {k: v.clone().detach() if isinstance(v, torch.Tensor) else v for k, v in self.current.items()}
                )

            self.current['theta'].detach_()
            self.current['target_val'].detach_()
            self.current['grad_val'].detach_()
