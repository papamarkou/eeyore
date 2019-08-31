import numpy as np

import torch

from eeyore.api import SerialSampler
from eeyore.mcmc import MCChain

class SMMALANP(SerialSampler):
    def __init__(self, model, theta0, dataloader, step=0.1, transform=None, keys=['theta', 'target_val', 'accepted']):
        super(SMMALANP, self).__init__()
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

        self.current['theta'] = np.copy(theta)
        self.current['target_val'], self.current['grad_val'], self.current['metric_val'] = \
            self.model.upto_metric_log_target(torch.from_numpy(self.current['theta']), data, label)
        self.current['target_val'] = self.current['target_val'].detach().cpu().numpy()
        self.current['grad_val'] = self.current['grad_val'].detach().cpu().numpy()
        self.current['metric_val'] = self.current['metric_val'].detach().cpu().numpy()
        if self.transform is not None:
            self.current['metric_val'] = self.transform(self.current['metric_val'])
        self.current['inv_metric_val'] = np.linalg.inv(self.current['metric_val'])
        self.current['chol_inv_metric_val'] = np.linalg.cholesky(self.current['inv_metric_val'])

        # See 'Riemann manifold Langevin and Hamiltonian Monte Carlo methods'
        # First summand appearing in equation (10) of page 130
        # Product of metric tensor with gradient
        self.current['first_term_val'] = np.matmul(self.current['inv_metric_val'], self.current['grad_val'])

    def draw(self, randn_val, threshold, savestate=False):
        proposed = {key : None for key in self.keys}

        for data, label in self.dataloader:
            proposal_mean = self.current['theta'] + 0.5 * self.step * self.current['first_term_val']

            proposed['theta'] = proposal_mean + np.sqrt(self.step) * \
                np.matmul(self.current['chol_inv_metric_val'], randn_val)
            proposed['target_val'], proposed['grad_val'], proposed['metric_val'] = \
                self.model.upto_metric_log_target(torch.from_numpy(proposed['theta']), data, label)
            proposed['target_val'] = proposed['target_val'].detach().cpu().numpy()
            proposed['grad_val'] = proposed['grad_val'].detach().cpu().numpy()
            proposed['metric_val'] = proposed['metric_val'].detach().cpu().numpy()
            if self.transform is not None:
                proposed['metric_val'] = self.transform(proposed['metric_val'])
            proposed['inv_metric_val'] = np.linalg.inv(proposed['metric_val'])
            proposed['first_term_val'] = np.matmul(proposed['inv_metric_val'], proposed['grad_val'])

            log_rate = proposed['target_val'] - self.current['target_val']
            loc_minus_proposal_mean = proposed['theta'] - proposal_mean
            inv_metric_sign, inv_metric_logdet = np.linalg.slogdet(self.step * self.current['inv_metric_val'])
            log_rate += \
                0.5 * (inv_metric_logdet + np.inner(loc_minus_proposal_mean, \
                np.matmul(self.current['metric_val'], loc_minus_proposal_mean))/self.step)

            proposal_mean = proposed['theta'] + 0.5 * self.step * proposed['first_term_val']
            loc_minus_proposal_mean = self.current['theta'] - proposal_mean
            inv_metric_sign, inv_metric_logdet = np.linalg.slogdet(self.step * proposed['inv_metric_val'])
            log_rate -= \
                0.5 * (inv_metric_logdet + np.inner(loc_minus_proposal_mean, \
                np.matmul(proposed['metric_val'], loc_minus_proposal_mean))/self.step)

            if np.log(threshold) < log_rate:
                self.current['theta'] = np.copy(proposed['theta'])
                self.current['target_val'] = np.copy(proposed['target_val'])
                self.current['grad_val'] = np.copy(proposed['grad_val'])
                self.current['metric_val'] = np.copy(proposed['metric_val'])
                self.current['inv_metric_val'] = np.copy(proposed['inv_metric_val'])
                self.current['chol_inv_metric_val'] = np.linalg.cholesky(proposed['inv_metric_val'])
                self.current['first_term_val'] = np.copy(proposed['first_term_val'])
                self.current['accepted'] = 1
            else:
                self.model.set_params(torch.from_numpy(self.current['theta']))
                self.current['accepted'] = 0

            if savestate:
                self.chain.update(self.current)

            return proposed, log_rate, self.current['accepted']
