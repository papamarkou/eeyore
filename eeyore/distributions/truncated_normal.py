# https://link.springer.com/article/10.1007/BF00143942

from torch.distributions import Exponential, Normal

from .truncated_distribution import TruncatedDistribution

class TruncatedNormal(TruncatedDistribution):
    def __init__(self, loc, scale, lower_bound=-float('inf'), upper_bound=float('inf'), *args, **kwargs):
        super().__init__(Normal(loc, scale), lower_bound=lower_bound, upper_bound=upper_bound, *args, **kwargs)

    def sample_lower_bounded_truncnorm():
        l = (self.lower_bound - self.base_dist.loc) / self.base_dist.scale
        rate = 0.5 * (l + (l ** 2 + 4).sqrt())

        while True:
            sample = Exponential(rate / self.base_dist.scale).sample() + self.lower_bound
            ratio = (-0.5 * ((sample - self.base_dist.loc) / self.base_dist.scale - rate) ** 2).exp()
            if torch.rand(1, dtype=self.base_dist.loc.dtype, device=self.base_dist.loc.device) <= ratio:
                break

        return sample
