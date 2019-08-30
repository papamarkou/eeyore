import torch


class TransitionKernel:
    """ Transition kernel for Metropolis Hastings. """

    def __init__(self, dtype=torch.float64, device='cpu'):
        self.dtype = dtype
        self.device = device

    def default_density(self):
        raise NotImplementedError

    def set_density(self):
        raise NotImplementedError

    def log_density(self, state):
        return torch.sum(self.density.log_prob(state), dtype=self.dtype)

    def sample(self):
        """ Sample the probability density function """
        return self.density.sample()
