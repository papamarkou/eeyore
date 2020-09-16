# https://link.springer.com/article/10.1007/BF00143942

import torch

from torch.distributions import Exponential, Normal, Uniform

from .truncated_distribution import TruncatedDistribution

class TruncatedNormal(TruncatedDistribution):
    def __init__(self, loc, scale, lower_bound=-float('inf'), upper_bound=float('inf'), *args, **kwargs):
        super().__init__(Normal(loc, scale), lower_bound=lower_bound, upper_bound=upper_bound, *args, **kwargs)

        self.a = (self.lower_bound - self.base_dist.loc) / self.base_dist.scale
        self.b = (self.upper_bound - self.base_dist.loc) / self.base_dist.scale

    def sample_lower_bounded(self):
        rate = 0.5 * (self.a + (self.a ** 2 + 4).sqrt())

        while True:
            sample = Exponential(rate / self.base_dist.scale).sample() + self.lower_bound
            ratio = (-0.5 * ((sample - self.base_dist.loc) / self.base_dist.scale - rate) ** 2).exp()
            if torch.rand(1, dtype=self.base_dist.loc.dtype, device=self.base_dist.loc.device) <= ratio:
                break

        return sample

    def sample_upper_bounded(self):
        rate = 0.5 * (-self.b + (self.b ** 2 + 4).sqrt())

        while True:
            sample = Exponential(rate / self.base_dist.scale).sample() - self.upper_bound
            ratio = (-0.5 * ((sample + self.base_dist.loc) / self.base_dist.scale - rate) ** 2).exp()
            if torch.rand(1, dtype=self.base_dist.loc.dtype, device=self.base_dist.loc.device) <= ratio:
                break

        return -sample

    def sample_doubly_bounded(self):
        while True:
            sample = Uniform(self.a, self.b).sample()

            if ((self.a < 0) and (0 < self.b)):
                ratio = (-0.5 * (sample ** 2)).exp()
            elif self.b < 0:
                ratio = (0.5 * (b ** 2 - sample ** 2)).exp()
            elif 0 < self.a:
                ratio = (0.5 * (a ** 2 - sample ** 2)).exp()

            if torch.rand(1, dtype=self.base_dist.loc.dtype, device=self.base_dist.loc.device) <= ratio:
                break

        return self.base_dist.loc + self.base_dist.scale * sample

    def sample(self):
        if ((self.lower_bound != -float('inf')) and (self.upper_bound != float('inf'))):
            return self.sample_doubly_bounded()
        elif ((self.lower_bound != -float('inf')) and (self.upper_bound == float('inf'))):
            return self.sample_lower_bounded()
        elif ((self.lower_bound == -float('inf')) and (self.upper_bound != float('inf'))):
            return self.sample_upper_bounded()
