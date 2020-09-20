# https://link.springer.com/article/10.1007/BF00143942

import torch

from scipy.stats import truncnorm

from torch.distributions import Exponential, Normal, Uniform

from .truncated_distribution import TruncatedDistribution

class TruncatedNormal(TruncatedDistribution):
    def __init__(self, loc, scale, lower_bound=-float('inf'), upper_bound=float('inf'), *args, **kwargs):
        super().__init__(Normal(loc, scale), lower_bound=lower_bound, upper_bound=upper_bound, *args, **kwargs)

        self.set_a_b()

    def set_a(self):
        self.a = (self.lower_bound - self.base_dist.loc) / self.base_dist.scale

    def set_b(self):
        self.b = (self.upper_bound - self.base_dist.loc) / self.base_dist.scale

    def set_a_b(self):
        self.set_a()
        self.set_b()

    def num_params(self):
        return len(self.base_dist.loc)

    def log_prob(self, x):
        return torch.sum(torch.from_numpy(truncnorm.logpdf(
            x.detach().cpu().numpy(),
            a=self.a.detach().cpu().numpy(),
            b=self.b.detach().cpu().numpy(),
            loc=self.base_dist.loc.detach().cpu().numpy(),
            scale=self.base_dist.scale.detach().cpu().numpy()
            )).to(dtype=self.base_dist.loc.dtype, device=self.base_dist.loc.device))

    def sample_lower_bounded(self):
        rate = 0.5 * (self.a + (self.a ** 2 + 4).sqrt())

        sample = []

        for i in range(self.num_params()):
            while True:
                proposed = Exponential(rate[i] / self.base_dist.scale[i]).sample() + self.lower_bound
                ratio = (-0.5 * ((proposed - self.base_dist.loc[i]) / self.base_dist.scale[i] - rate[i]) ** 2).exp()
                if torch.rand(1, dtype=self.base_dist.loc.dtype, device=self.base_dist.loc.device) <= ratio:
                    break

            sample.append(proposed)

        return torch.stack(sample)

    def sample_upper_bounded(self):
        rate = 0.5 * (-self.b + (self.b ** 2 + 4).sqrt())

        sample = []

        for i in range(self.num_params()):
            while True:
                proposed = Exponential(rate[i] / self.base_dist.scale[i]).sample() - self.upper_bound
                ratio = (-0.5 * ((proposed + self.base_dist.loc[i]) / self.base_dist.scale[i] - rate[i]) ** 2).exp()
                if torch.rand(1, dtype=self.base_dist.loc.dtype, device=self.base_dist.loc.device) <= ratio:
                    break

            sample.append(-proposed)

        return torch.stack(sample)

    def sample_doubly_bounded(self):
        sample = []

        for i in range(self.num_params()):
            while True:
                proposed = Uniform(self.a[i], self.b[i]).sample()

                if ((self.a[i] < 0) and (0 < self.b[i])):
                    ratio = (-0.5 * (proposed ** 2)).exp()
                elif self.b[i] < 0:
                    ratio = (0.5 * (self.b[i] ** 2 - proposed ** 2)).exp()
                elif 0 < self.a[i]:
                    ratio = (0.5 * (self.a[i] ** 2 - proposed ** 2)).exp()

                if torch.rand(1, dtype=self.base_dist.loc.dtype, device=self.base_dist.loc.device) <= ratio:
                    break

            sample.append(proposed)

        return self.base_dist.loc + self.base_dist.scale * torch.stack(sample)

    def sample(self):
        if ((self.lower_bound != -float('inf')) and (self.upper_bound != float('inf'))):
            return self.sample_doubly_bounded()
        elif ((self.lower_bound != -float('inf')) and (self.upper_bound == float('inf'))):
            return self.sample_lower_bounded()
        elif ((self.lower_bound == -float('inf')) and (self.upper_bound != float('inf'))):
            return self.sample_upper_bounded()
