import torch
from torch.distributions import Normal

from eeyore.api import Kernel

class NormalKernel(Kernel):
    """ Gaussian distributed transition kernel """

    def __init__(self, mu, sigma, dtype=torch.float64, device='cpu'):
        super(NormalKernel, self).__init__(dtype=dtype, device=device)
        self.density = Normal(mu.to(self.dtype).to(self.device), sigma.to(self.dtype).to(self.device))

    def set_density(self, mu):
        """ Set the probability density function """
        self.density = Normal(mu.to(self.dtype).to(self.device), self.density.scale)
