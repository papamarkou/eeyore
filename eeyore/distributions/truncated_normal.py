from .truncated_distribution import TruncatedDistribution

class TruncatedNormal(TruncatedDistribution):
    def __init__(self, base_dist, lower_bound=-float('inf'), upper_bound=float('inf'), *args, **kwargs):
        super().__init__(base_dist, lower_bound=lower_bound, upper_bound=upper_bound, *args, **kwargs)
