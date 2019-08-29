import torch
from torch.distributions import Normal

from eeyore.api import TransitionKernel

class NormalTransitionKernel(TransitionKernel):
    """ Gaussian Distributed Transition Kernel """

    def __init__(self, mu, sigma):
        super(NormalTransitionKernel, self).__init__()
        self.density = Normal(mu, sigma)

    def set_density(self, mu):
        """ Set the probability density function """
        self.density = Normal(mu, self.density.scale)
