import torch

from .homogeneous_kernel import HomogeneousKernel

class IsoSEKernel(HomogeneousKernel):
    """ Isotropic squared exponential kernel """

    def __init__(self, sigma=1., l=1.):
        self.sigma = sigma # square of amplitude
        self.l = l # square of lengthscale

    def k(self, x1, x2):
        return  torch.exp(-self.squared_dist(x1, x2).div(2. * self.l)).mul(self.sigma)
