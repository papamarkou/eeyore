import torch

from .homogeneous_kernel import HomogeneousKernel

class PeriodicKernel(HomogeneousKernel):
    """ Periodic kernel """

    def __init__(self, sigma=1., l=1., p=2.):
        self.sigma = sigma # square of amplitude
        self.l = l # square or lengthscale
        self.p = p # multiple of period

    def k(self, x1, x2):
        return torch.exp(-torch.sin(self.dist(x1, x2).div(self.p)).pow(2).mul(2.).div(self.l)).mul(self.sigma)
