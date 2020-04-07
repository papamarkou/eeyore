import torch

from .kernel import Kernel

class HomogeneousKernel(Kernel):
    """ Base class for function kernels based on distance between inputs """

    def dist(self, x1, x2):
        return torch.norm(x1 - x2, 2)

    def squared_dist(self, x1, x2):
        return self.dist(x1, x2).pow(2)
