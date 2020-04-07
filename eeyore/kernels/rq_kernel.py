import torch

from .homogeneous_kernel import HomogeneousKernel

class RQKernel(HomogeneousKernel):
    """ Rational quadratic kernel """

    def __init__(self, sigma=1., l=1., a=1.):
        self.sigma = sigma # square of amplitude
        self.l = l # square of lengthscale
        self.a = a # scale mixture (a > 0)

    def k(self, x1, x2):
        return  self.squared_dist(x1, x2).div(2. * self.a * self.l).add(1.).pow(-self.a).mul(self.sigma)
