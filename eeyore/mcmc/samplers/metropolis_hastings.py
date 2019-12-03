import torch

from eeyore.api import SerialSampler
from eeyore.kernels import NormalKernel
from eeyore.mcmc import ChainFile, ChainList

class MetropolisHastings(SerialSampler):
    def __init__(self, model, theta0, dataloader,
        symmetric=True, kernel=None, chain=ChainList(keys=['theta', 'target_val', 'accepted'])):
        super(MetropolisHastings, self).__init__()
        self.model = model
        self.dataloader = dataloader
        self.symmetric = symmetric

        self.kernel = kernel or self.default_kernel(theta0.clone().detach())
        self.keys = ['theta', 'target_val', 'accepted']
        self.current = {key : None for key in self.keys}
        self.chain = chain

        self.reset(theta0)

    def default_kernel(self, theta):
        return NormalKernel(theta, torch.ones(self.model.num_params()))

    def reset(self, theta):
        self.current['theta'] = theta.clone().detach()
        self.current['target_val'] = self.model.log_target(self.current['theta'].clone().detach(), self.dataloader)
        self.kernel.set_density_params(self.current['theta'].clone().detach())

    def draw(self, n, savestate=False):
        proposed = {key : None for key in self.keys}

        proposed['theta'] = self.kernel.sample()
        proposed['target_val'] = self.model.log_target(proposed['theta'].clone().detach(), self.dataloader)

        log_rate = proposed['target_val'] - self.current['target_val']
        if not self.symmetric:
            log_rate = log_rate - self.kernel.log_density(proposed['theta'].clone().detach())
            self.kernel.set_density_params(proposed['theta'].clone().detach())
            log_rate = log_rate + self.kernel.log_density(self.current['theta'].clone().detach())

        if torch.log(torch.rand(1, dtype=self.model.dtype, device=self.model.device)) < log_rate:
            self.current['theta'] = proposed['theta'].clone().detach()
            self.current['target_val'] = proposed['target_val'].clone().detach()
            if self.symmetric:
                self.kernel.set_density_params(proposed['theta'].clone().detach())
            self.current['accepted'] = 1
        else:
            self.model.set_params(self.current['theta'].clone().detach())
            if not self.symmetric:
                self.kernel.set_density_params(self.current['theta'].clone().detach())
            self.current['accepted'] = 0

        if savestate:
            self.chain.update(
                {k: v.clone().detach() if isinstance(v, torch.Tensor) else v for k, v in self.current.items()}
            )

        self.current['theta'].detach_()
        self.current['target_val'].detach_()
