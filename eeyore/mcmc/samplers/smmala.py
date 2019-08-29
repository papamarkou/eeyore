import numpy as np

import torch

from eeyore.api import SerialSampler
from eeyore.mcmc import MCChain

class SMMALA(SerialSampler):
    def __init__(self, model, theta0, dataloader, step=0.1, transform=None, keys=['theta', 'target_val', 'accepted']):
        super(SMMALA, self).__init__()
        self.model = model
        self.dataloader = dataloader
        self.step = step
        self.transform = transform

        self.keys = ['theta', 'target_val', 'grad_val', 'metric_val', 'inv_metric_val', 'first_term_val']
        self.current = {key : None for key in self.keys}
        self.chain = MCChain(keys)

        self.reset(theta0)

    def reset(self, theta):
        data, label = next(iter(self.dataloader))

        self.current['theta'] = theta.clone().detach()
        self.current['theta'].requires_grad_(True)
        self.current['target_val'], self.current['grad_val'], self.current['metric_val'] = \
            self.model.upto_metric_log_target(self.current['theta'].clone().detach(), data, label)
        if self.transform is not None:
            self.current['metric_val'] = self.transform(self.current['metric_val'])
        self.current['inv_metric_val'] = torch.inverse(self.current['metric_val'])
        self.current['chol_inv_metric_val'] = torch.cholesky(self.current['inv_metric_val'])

        # See 'Riemann manifold Langevin and Hamiltonian Monte Carlo methods'
        # First summand appearing in equation (10) of page 130
        # Product of metric tensor with gradient
        self.current['first_term_val'] = self.current['inv_metric_val'] @ self.current['grad_val']

    def draw(self, savestate=False):
        proposed = {key : None for key in self.keys}

        for data, label in self.dataloader:
            proposal_mean = self.current['theta'] + 0.5 * self.step * self.current['first_term_val']

            proposed['theta'] = \
                proposal_mean + np.sqrt(self.step) * self.current['chol_inv_metric_val'] @ \
                torch.randn(self.model.num_params(), dtype=self.model.dtype)
            proposed['target_val'], proposed['grad_val'], proposed['metric_val'] = \
                self.model.upto_metric_log_target(proposed['theta'].clone().detach(), data, label)
            if self.transform is not None:
                proposed['metric_val'] = self.transform(proposed['metric_val'])
            proposed['inv_metric_val'] = torch.inverse(proposed['metric_val'])
            proposed['first_term_val'] = proposed['inv_metric_val'] @ proposed['grad_val']

            log_rate = proposed['target_val'] - self.current['target_val']
            loc_minus_proposal_mean = proposed['theta'] - proposal_mean
            inv_metric_sign, inv_metric_logdet = torch.slogdet(self.step * self.current['inv_metric_val'])
            log_rate = \
                log_rate + 0.5 * (inv_metric_logdet + (loc_minus_proposal_mean.t() @ (self.current['metric_val'] @ \
                loc_minus_proposal_mean))/self.step)

            proposal_mean = proposed['theta'] + 0.5 * self.step * proposed['first_term_val']

            loc_minus_proposal_mean = self.current['theta'] - proposal_mean
            inv_metric_sign, inv_metric_logdet = torch.slogdet(self.step * proposed['inv_metric_val'])
            log_rate = \
                log_rate - 0.5 * (inv_metric_logdet +(loc_minus_proposal_mean.t() @ (proposed['metric_val'] @ \
                loc_minus_proposal_mean))/self.step)

            if torch.log(torch.rand(1, dtype=self.model.dtype)) < log_rate:
                self.current['theta'] = proposed['theta'].clone().detach()
                self.current['target_val'] = proposed['target_val'].clone().detach()
                self.current['grad_val'] = proposed['grad_val'].clone().detach()
                self.current['metric_val'] = proposed['metric_val'].clone().detach()
                self.current['inv_metric_val'] = proposed['inv_metric_val'].clone().detach()
                self.current['chol_inv_metric_val'] = torch.cholesky(self.current['inv_metric_val']).clone().detach()
                self.current['first_term_val'] = proposed['first_term_val'].clone().detach()
                self.current['accepted'] = 1
            else:
                self.model.set_params(self.current['theta'].clone().detach())
                self.current['accepted'] = 0

            if savestate:
                self.chain.update(
                    {k: v.clone().detach() if isinstance(v, torch.Tensor) else v for k, v in self.current.items()})

            self.current['theta'].detach_()
            self.current['target_val'].detach_()
            self.current['grad_val'].detach_()
            self.current['metric_val'].detach_()
            self.current['inv_metric_val'].detach_()
            self.current['chol_inv_metric_val'].detach_()
            self.current['first_term_val'].detach_()
