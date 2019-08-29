import torch

from eeyore.api import SerialSampler
from eeyore.kernels import NormalTransitionKernel
from eeyore.mcmc import MCChain

class MetropolisHastings(SerialSampler):
    def __init__(self, model, theta0, dataloader, kernel=None, keys=['theta', 'target_val', 'accepted']):
        super(MetropolisHastings, self).__init__()
        self.model = model
        self.dataloader = dataloader

        self.kernel = kernel or self.default_kernel()
        self.keys = ['theta', 'target_val']
        self.current = {key : None for key in self.keys}
        self.chain = MCChain(keys)

        self.reset(theta0)

    def default_kernel(self):
        return NormalTransitionKernel(
            torch.zeros(self.model.num_params(), dtype=self.model.dtype),
            torch.ones(self.model.num_params(), dtype=self.model.dtype)
        )

    def reset(self, theta):
        data, label = next(iter(self.dataloader))

        self.current['theta'] = theta.clone().detach()
        self.current['target_val'] = self.model.log_target(self.current['theta'].clone().detach(), data, label)

    def draw(self, savestate=False):
        proposed = {key : None for key in self.keys}

        for data, label in self.dataloader:
            proposed['theta'] = self.kernel.sample()
            proposed['target_val'] = self.model.log_target(proposed['theta'], data, label)

            log_rate = - self.current['target_val'] - self.kernel.log_density(proposed['theta'].clone().detach())

            self.kernel.set_density(proposed['theta'].clone().detach())

            log_rate = \
                log_rate + proposed['target_val'] + self.kernel.log_density(self.current['theta'].clone().detach())

            if torch.log(torch.rand(1, dtype=self.model.dtype)) < log_rate:
                self.current['theta'] = proposed['theta'].clone().detach()
                self.current['target_val'] = proposed['target_val'].clone().detach()
                self.current['accepted'] = 1
            else:
                self.model.set_params(self.current['theta'].clone().detach())
                self.kernel.set_density(self.current['theta'].clone().detach())
                self.current['accepted'] = 0

            if savestate:
                self.chain.update(
                    {k: v.clone().detach() if isinstance(v, torch.Tensor) else v for k, v in self.current.items()}
                )

            self.current['theta'].detach_()
            self.current['target_val'].detach_()
