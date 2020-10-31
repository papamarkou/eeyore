# https://github.com/pytorch/pytorch/pull/32377/commits/d4dae5c2d1fbb0f23c69a40bbad2e6dfdf2fa6b7

from torch.distributions.distribution import Distribution

class TruncatedDistribution(Distribution):
    def __init__(self, base_dist, lower_bound=-float('inf'), upper_bound=float('inf'), *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.base_dist = base_dist
        self.z = self.base_dist.cdf(self.upper_bound) - self.base_dist.cdf(self.lower_bound)

    def log_prob(self, x):
        log_prob_val = self.base_dist.log_prob(x) - self.z.log()
        log_prob_val[(x <= self.lower_bound) | (x >= self.upper_bound)] = -float('inf')
        return log_prob_val

    def cdf(self, x):
        return (self.base_dist.cdf(x) - self.base_dist.cdf(self.lower_bound)) / self.z
