from torch.distributions import MultivariateNormal

from .normalized_kernel import NormalizedKernel

class MultivariateNormalKernel(NormalizedKernel):
    """ Multivariate normal kernel """

    def __init__(self, mu, scale_tril):
        self.set_density(mu, scale_tril)

    def set_density(self, mu, scale_tril):
        """ Set multivariate normal probability density function """
        self.density = MultivariateNormal(mu, scale_tril=scale_tril)

    def set_density_params(self, mu, scale_tril=None):
        """ Set the parameters of multivariate normal probability density function """
        self.density.loc = mu
        if scale_tril is not None:
            self.density.scale_tril = scale_tril

    def k(self, x1, x2, sigma=None):
        self.set_density_params(x2, sigma=sigma)
        return self.log_density(x1).exp()
