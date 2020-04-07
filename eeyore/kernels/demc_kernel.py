import torch

from torch.distributions import Normal

from .normalized_kernel import NormalizedKernel

class DEMCKernel(NormalizedKernel):
    """ Normal transition kernel for DEMC"""

    def __init__(self, a=None, b=None, c=0.1, density=None):
        self.a = a
        self.b = b
        self.c = c
        self.density = density

    def init_a_and_b(self, n, dtype, device):
        self.a = torch.empty(n, dtype=dtype, device=device)
        self.b = torch.empty(n, dtype=dtype, device=device)

    def init_density(self, n, dtype, device):
        self.density = Normal(torch.empty(n, dtype=dtype, device=device), torch.empty(n, dtype=dtype, device=device))

    def set_a_and_b(self, a, b):
        self.a = a
        self.b = b

    def mean(self, theta):
        return theta + self.c * (self.a - self.b)

    def set_density(self, theta, sigma):
        """ Set normal probability density function """
        self.density = Normal(self.mean(theta), sigma)

    def set_density_params(self, theta, sigma=None):
        """ Set the parameters of of normal probability density function """
        self.density.loc = self.mean(theta)
        if sigma is not None:
            self.density.scale = sigma
