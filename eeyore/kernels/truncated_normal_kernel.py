from eeyore.distributions import TruncatedNormal

from .normalized_kernel import NormalizedKernel

class TruncatedNormalKernel(NormalizedKernel):
    """ Truncated normal kernel """

    def __init__(self, loc, scale, lower_bound=-float('inf'), upper_bound=float('inf')):
        self.set_density(loc, scale, lower_bound, upper_bound)

    def set_density(self, loc, scale, lower_bound=-float('inf'), upper_bound=float('inf')):
        """ Set truncated normal probability density function """
        self.density = TruncatedNormal(loc, scale, lower_bound=lower_bound, upper_bound=upper_bound)

    def set_density_params(self, loc, scale=None, lower_bound=-float('inf'), upper_bound=float('inf')):
        """ Set the parameters of truncated normal probability density function """
        self.density.base_dist.loc = loc
        if scale is not None:
            self.density.base_dist.scale = scale
        self.density.set_a_b()
