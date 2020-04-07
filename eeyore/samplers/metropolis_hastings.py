import torch

from .serial_sampler import SerialSampler
from eeyore.chains import ChainList
from eeyore.datasets import DataCounter
from eeyore.kernels import NormalKernel

class MetropolisHastings(SerialSampler):
    def __init__(self, model, theta0,
        dataloader=None, data0=None, counter=None,
        symmetric=True, kernel=None, chain=ChainList(keys=['sample', 'target_val', 'accepted'])):
        super(MetropolisHastings, self).__init__(counter or DataCounter.from_dataloader(dataloader))
        self.model = model
        self.dataloader = dataloader
        self.symmetric = symmetric

        self.kernel = kernel or self.default_kernel(theta0.clone().detach())
        self.keys = ['sample', 'target_val', 'accepted']
        self.current = {key : None for key in self.keys}
        self.chain = chain

        x, y = data0 or next(iter(self.dataloader))
        self.reset(theta0.clone().detach(), x, y)

    def default_kernel(self, theta):
        return NormalKernel(theta, torch.ones(self.model.num_params()))

    def reset(self, theta, x, y, sigma=None, scale_tril=None):
        self.current['sample'] = theta
        self.current['target_val'] = self.model.log_target(self.current['sample'].clone().detach(), x, y)
        if sigma is not None:
            self.kernel.set_density_params(self.current['sample'].clone().detach(), sigma=sigma)
        elif scale_tril is not None:
            self.kernel.set_density_params(
                self.current['sample'].clone().detach(), scale_tril=scale_tril.clone().detach()
            )
        else:
            self.kernel.set_density_params(self.current['sample'].clone().detach())

    def draw(self, x, y, savestate=False):
        proposed = {key : None for key in self.keys}

        proposed['sample'] = self.kernel.sample()
        proposed['target_val'] = self.model.log_target(proposed['sample'].clone().detach(), x, y)

        log_rate = proposed['target_val'] - self.current['target_val']
        if not self.symmetric:
            log_rate = log_rate - self.kernel.log_density(proposed['sample'].clone().detach())
            self.kernel.set_density_params(proposed['sample'].clone().detach())
            log_rate = log_rate + self.kernel.log_density(self.current['sample'].clone().detach())

        if torch.log(torch.rand(1, dtype=self.model.dtype, device=self.model.device)) < log_rate:
            self.current['sample'] = proposed['sample'].clone().detach()
            self.current['target_val'] = proposed['target_val'].clone().detach()
            if self.symmetric:
                self.kernel.set_density_params(proposed['sample'].clone().detach())
            self.current['accepted'] = 1
        else:
            self.model.set_params(self.current['sample'].clone().detach())
            if not self.symmetric:
                self.kernel.set_density_params(self.current['sample'].clone().detach())
            self.current['accepted'] = 0

        if savestate:
            self.chain.detach_and_update(self.current)

        self.current['sample'].detach_()
        self.current['target_val'].detach_()