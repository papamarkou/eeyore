import numpy as np

import torch

from torch.distributions import Normal

from eeyore.api import SerialSampler
from eeyore.mcmc import MCChain


class MALA(SerialSampler):

    def __init__(self, model, theta0, dataloader, step=0.1, keys=['theta', 'target_val', 'accepted']):
        super(MALA, self).__init__()
        self.model = model
        self.dataloader = dataloader
        self.step = step

        self.keys = ['theta', 'target_val', 'grad_val']
        self.current = {key : None for key in self.keys}
        self.chain = MCChain(keys)

        self.reset(theta0)

    def reset(self, theta):
        data, label = next(iter(self.dataloader))

        self.current['theta'] = theta.detach()
        self.current['target_val'], self.current['grad_val'] = \
            self.model.upto_grad_log_target(self.current['theta'], data, label)

    def draw(self, savestate=False):
        proposed = {key : None for key in self.keys}

        for data, label in self.dataloader:
            proposal_mean = self.current['theta'] + 0.5 * self.step * self.current['grad_val']

            proposed['theta'] = Normal(proposal_mean, np.sqrt(self.step)).sample()
            proposed['target_val'], proposed['grad_val'] = \
                self.model.upto_grad_log_target(proposed['theta'], data, label)

            log_rate = proposed['target_val'] - self.current['target_val']
            log_rate = log_rate + 0.5 * torch.sum((proposed['theta'] - proposal_mean) ** 2) / self.step

            proposal_mean = proposed['theta'] + 0.5 * self.step * proposed['grad_val']

            log_rate = log_rate - 0.5 * torch.sum((self.current['theta'] - proposal_mean) ** 2) / self.step

            threshold = torch.rand(1)
            if torch.log(threshold) < log_rate:
                self.current['theta'] = proposed['theta'].detach()
                self.current['target_val'] = proposed['target_val'].detach()
                self.current['grad_val'] = proposed['grad_val'].detach()
                self.current['accepted'] = 1
            else:
                self.model.set_params(self.current['theta'])
                self.current['accepted'] = 0

            if savestate:
                self.chain.update(self.current)
