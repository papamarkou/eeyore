from torch.distributions import MultivariateNormal

from .normalized_kernel import NormalizedKernel

class MultivariateNormalKernel(NormalizedKernel):
    """ Multivariate normal kernel """

    def __init__(self, loc, scale_tril):
        self.set_density(loc, scale_tril)

    def set_density(self, loc, scale_tril):
        """ Set multivariate normal probability density function """
        self.density = MultivariateNormal(loc, scale_tril=scale_tril)

    def set_density_params(self, loc, scale_tril=None):
        """ Set the parameters of multivariate normal probability density function """
        self.density.loc = loc
        if scale_tril is not None:
            self.density.scale_tril = scale_tril

    def k(self, x1, x2, scale=None):
        self.set_density_params(x2, scale=scale)
        return self.log_prob(x1).exp()
