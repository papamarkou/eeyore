import torch

from eeyore.api import SerialSampler
from eeyore.kernels import MultivariateNormalKernel
from eeyore.mcmc import ChainFile, ChainList

class RAM(SerialSampler):
    def __init__(
        self, model, theta0, dataloader,
        s0=None, a=0.234, g=0.7, chain=ChainList(keys=['theta', 'target_val', 'accepted'])):
        super(RAM, self).__init__()
        self.model = model
        self.dataloader = dataloader
        self.a = a
        self.g = g

        self.s = s0 or torch.eye(self.model.num_params(), dtype=self.model.dtype, device=self.model.device)
        self.keys = ['theta', 'target_val', 'accepted']
        self.current = {key : None for key in self.keys}
        self.chain = chain

        self.reset(theta0)

    def default_kernel(self, theta):
        return MultivariateNormalKernel(theta, torch.eye(self.model.num_params()))

    def reset(self, theta, s=None):
        self.current['theta'] = theta.clone().detach()
        self.current['target_val'] = self.model.log_target(self.current['theta'].clone().detach(), self.dataloader)
        if s is not None:
            self.s = s

    def draw(self, n, savestate=False):
        proposed = {key : None for key in self.keys}

        randn_sample = torch.randn(self.model.num_params(), dtype=self.model.dtype, device=self.model.device)
        proposed['theta'] = self.current['theta'].clone().detach() + self.s @ randn_sample
        proposed['target_val'] = self.model.log_target(proposed['theta'].clone().detach(), self.dataloader)

        log_rate = proposed['target_val'] - self.current['target_val']

        if torch.log(torch.rand(1, dtype=self.model.dtype, device=self.model.device)) < log_rate:
            self.current['theta'] = proposed['theta'].clone().detach()
            self.current['target_val'] = proposed['target_val'].clone().detach()
            self.current['accepted'] = 1
        else:
            self.model.set_params(self.current['theta'].clone().detach())
            self.current['accepted'] = 0

        h = min(1, self.model.num_params() * (n + 1) ** (-self.g))
        self.s = torch.cholesky(self.s @ (
            torch.eye(self.model.num_params(), dtype=self.model.dtype, device=self.model.device) + \
            h * (min(1, torch.exp(log_rate).item()) - self.a
            ) * torch.ger(randn_sample, randn_sample) / torch.dot(randn_sample, randn_sample).item()) @ self.s.t())

        if savestate:
            self.chain.update(
                {k: v.clone().detach() if isinstance(v, torch.Tensor) else v for k, v in self.current.items()}
            )

        self.current['theta'].detach_()
        self.current['target_val'].detach_()
