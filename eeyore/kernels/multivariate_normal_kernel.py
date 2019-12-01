import torch
from torch.distributions import MultivariateNormal

from eeyore.api import Kernel

class MultivariateNormalKernel(Kernel):
    """ Multivariate normal transition kernel """

    def __init__(self, mu, scale_tril):
        self.set_density(mu, scale_tril)

    def set_density(self, mu, scale_tril):
        """ Set multivariate normal probability density function """
        self.density = MultivariateNormal(mu, scale_tril)

    def set_density_params(self, mu, scale_tril=None):
        """ Set the parameters of multivariate normal probability density function """
        self.density.mu = mu
        if scale_tril is not None:
            self.density.scale_tril = scale_tril
