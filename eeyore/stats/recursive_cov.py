import torch

def recursive_cov(lastcov, lastmean, secondlastmean, n, x, offset=0):
    k = n - offset
    print("n =", n, ", offset =", offset, ", k =", k)
    return (
        (k - 1) * lastcov
        + torch.ger(x, x)
        - (k + 1) * torch.ger(lastmean, lastmean)
        + k * torch.ger(secondlastmean, secondlastmean)
        ) / k
