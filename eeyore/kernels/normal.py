import torch
from torch.distributions import Normal

from eeyore.api import TransitionKernel

class NormalTransitionKernel(TransitionKernel):
    """ Gaussian Distributed Transition Kernel """

    def __init__(self, mu, sigma, dtype=torch.float64, device='cpu'):
        super(NormalTransitionKernel, self).__init__(dtype=dtype, device=device)
        self.density = Normal(mu.to(self.dtype).to(self.device), sigma.to(self.dtype).to(self.device))

    def set_density(self, mu):
        """ Set the probability density function """
        self.density = Normal(mu.to(self.dtype).to(self.device), self.density.scale)
