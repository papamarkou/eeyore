import torch
from torch.distributions import Normal

from eeyore.api import Kernel

class DEMCKernel(Kernel):
    """ Gaussian distributed transition kernel """

    def __init__(self, theta, a, b, sigma, c=0.1, dtype=torch.float64, device='cpu'):
        super(DEMCKernel, self).__init__(dtype=dtype, device=device)
        self.a = a
        self.b = b
        self.c = c
        self.density = Normal(self.mean(theta).to(self.dtype).to(self.device), sigma.to(self.dtype).to(self.device))

    def mean(self, theta):
        return theta + self.c * (self.a - self.b)

    def set_density(self, theta):
        """ Set the probability density function """
        self.density = Normal(self.mean(theta).to(self.dtype).to(self.device), self.density.scale)
