import torch
from torch.distributions import Normal

from eeyore.api import Kernel

class NormalKernel(Kernel):
    """ Normal transition kernel """

    def __init__(self, mu, sigma):
        self.set_density(mu, sigma)

    def set_density(self, mu, sigma):
        """ Set normal probability density function """
        self.density = Normal(mu, sigma)

    def set_density_params(self, mu, sigma=None):
        """ Set the parameters of normal probability density function """
        self.density.loc = mu
        if sigma is not None:
            self.density.scale = sigma
