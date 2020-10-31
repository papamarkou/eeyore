from torch.distributions import Normal

from .normalized_kernel import NormalizedKernel

class NormalKernel(NormalizedKernel):
    """ Normal kernel """

    def __init__(self, loc, scale):
        self.set_density(loc, scale)

    def set_density(self, loc, scale):
        """ Set normal probability density function """
        self.density = Normal(loc, scale)

    def set_density_params(self, loc, scale=None):
        """ Set the parameters of normal probability density function """
        self.density.loc = loc
        if scale is not None:
            self.density.scale = scale

    def k(self, x1, x2, scale=None):
        self.set_density_params(x2, scale=scale)
        return self.log_prob(x1).exp()
