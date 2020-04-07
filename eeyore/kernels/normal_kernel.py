from torch.distributions import Normal

from .normalized_kernel import NormalizedKernel

class NormalKernel(NormalizedKernel):
    """ Normal kernel """

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

    def k(self, x1, x2, sigma=None):
        self.set_density_params(x2, sigma=sigma)
        return self.log_density(x1).exp()
