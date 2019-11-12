import torch
from torch.distributions import Normal

from eeyore.api import Kernel

class DEMCKernel(Kernel):
    """ Gaussian distributed transition kernel """

    def __init__(self, sigma, c=0.1, dtype=torch.float64, device='cpu'):
        super(DEMCKernel, self).__init__(dtype=dtype, device=device)
        self.a = None
        self.b = None
        self.sigma = sigma
        self.c = c
        self.density = None

    def set_a_and_b(self, a, b):
        self.a = a
        self.b = b

    def mean(self, theta):
        return theta + self.c * (self.a - self.b)

    def set_density(self, theta):
        """ Set the probability density function """
        self.density = Normal(self.mean(theta).to(self.dtype).to(self.device), self.sigma)
