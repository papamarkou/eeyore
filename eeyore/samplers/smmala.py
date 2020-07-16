import numpy as np
import torch

from .single_chain_serial_sampler import SingleChainSerialSampler
from eeyore.chains import ChainList
from eeyore.datasets import DataCounter

class SMMALA(SingleChainSerialSampler):
    def __init__(self, model, theta0=None, dataloader=None, data0=None, counter=None, step=0.1, transform=None,
    chain=ChainList()):
        super(SMMALA, self).__init__(counter or DataCounter.from_dataloader(dataloader))
        self.model = model
        self.dataloader = dataloader
        self.step = step
        self.transform = transform

        self.keys = ['sample', 'target_val', 'grad_val', 'metric_val', 'inv_metric_val', 'first_term_val', 'accepted']
        self.chain = chain

        if theta0 is not None:
            self.set_current(theta0.clone().detach(), data=data0)

    def set_current(self, theta, data=None):
        x, y = super().set_current(theta, data=data)
        self.current['sample'].requires_grad_(True)
        self.current['target_val'], self.current['grad_val'], self.current['metric_val'] = \
            self.model.upto_metric_log_target(self.current['sample'].clone().detach(), x, y)
        if self.transform is not None:
            self.current['metric_val'] = self.transform(self.current['metric_val'])
        self.current['inv_metric_val'] = torch.inverse(self.current['metric_val'])
        self.current['chol_inv_metric_val'] = torch.cholesky(self.current['inv_metric_val'])

        # See 'Riemann manifold Langevin and Hamiltonian Monte Carlo methods'
        # First summand appearing in equation (10) of page 130
        # Product of metric tensor with gradient
        self.current['first_term_val'] = self.current['inv_metric_val'] @ self.current['grad_val']

    def draw(self, x, y, savestate=False):
        proposed = {key : None for key in self.keys}

        proposal_mean = self.current['sample'] + 0.5 * self.step * self.current['first_term_val']

        proposed['sample'] = \
            proposal_mean + np.sqrt(self.step) * self.current['chol_inv_metric_val'] @ \
            torch.randn(self.model.num_params(), dtype=self.model.dtype, device=self.model.device)
        proposed['target_val'], proposed['grad_val'], proposed['metric_val'] = \
            self.model.upto_metric_log_target(proposed['sample'].clone().detach(), x, y)
        if self.transform is not None:
            proposed['metric_val'] = self.transform(proposed['metric_val'])
        proposed['inv_metric_val'] = torch.inverse(proposed['metric_val'])
        proposed['first_term_val'] = proposed['inv_metric_val'] @ proposed['grad_val']

        log_rate = proposed['target_val'] - self.current['target_val']
        loc_minus_proposal_mean = proposed['sample'] - proposal_mean
        inv_metric_sign, inv_metric_logdet = torch.slogdet(self.step * self.current['inv_metric_val'])
        log_rate = \
            log_rate + 0.5 * (inv_metric_logdet + (loc_minus_proposal_mean.t() @ (self.current['metric_val'] @ \
            loc_minus_proposal_mean))/self.step)

        proposal_mean = proposed['sample'] + 0.5 * self.step * proposed['first_term_val']

        loc_minus_proposal_mean = self.current['sample'] - proposal_mean
        inv_metric_sign, inv_metric_logdet = torch.slogdet(self.step * proposed['inv_metric_val'])
        log_rate = \
            log_rate - 0.5 * (inv_metric_logdet +(loc_minus_proposal_mean.t() @ (proposed['metric_val'] @ \
            loc_minus_proposal_mean))/self.step)

        if torch.log(torch.rand(1, dtype=self.model.dtype, device=self.model.device)) < log_rate:
            self.current['sample'] = proposed['sample'].clone().detach()
            self.current['target_val'] = proposed['target_val'].clone().detach()
            self.current['grad_val'] = proposed['grad_val'].clone().detach()
            self.current['metric_val'] = proposed['metric_val'].clone().detach()
            self.current['inv_metric_val'] = proposed['inv_metric_val'].clone().detach()
            self.current['chol_inv_metric_val'] = torch.cholesky(self.current['inv_metric_val']).clone().detach()
            self.current['first_term_val'] = proposed['first_term_val'].clone().detach()
            self.current['accepted'] = 1
        else:
            self.model.set_params(self.current['sample'].clone().detach())
            self.current['accepted'] = 0

        if savestate:
            self.chain.detach_and_update(self.current)

        self.current['sample'].detach_()
        self.current['target_val'].detach_()
        self.current['grad_val'].detach_()
        self.current['metric_val'].detach_()
        self.current['inv_metric_val'].detach_()
        self.current['chol_inv_metric_val'].detach_()
        self.current['first_term_val'].detach_()
