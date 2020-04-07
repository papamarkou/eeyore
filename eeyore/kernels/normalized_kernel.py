import torch

from .kernel import Kernel

class NormalizedKernel(Kernel):
    """ Base class for normalized kernels """

    def default_density(self):
        raise NotImplementedError

    def set_density(self):
        raise NotImplementedError

    def log_density(self, state):
        return torch.sum(self.density.log_prob(state))

    def sample(self):
        """ Sample the probability density function """
        return self.density.sample()
