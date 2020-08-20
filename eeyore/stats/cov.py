import torch

# https://discuss.pytorch.org/t/covariance-and-gradient-support/16217

def cov(x, rowvar=True):
    if x.dim() > 2:
        raise ValueError('x has more than 2 dimensions')
    if x.dim() < 2:
        x = x.view(1, -1)
    if not rowvar and x.size(0) != 1:
        x = x.t()

    x_ctr = x - torch.mean(x, dim=1, keepdim=True)

    return x_ctr.matmul(x_ctr.t()).squeeze() / (x.size(1) - 1)

def recursive_cov(lastcov, lastmean, secondlastmean, n, x, offset=0):
    k = n - offset
    print("n =", n, ", offset =", offset, ", k =", k)
    return (
        (k - 1) * lastcov
        + torch.ger(x, x)
        - (k + 1) * torch.ger(lastmean, lastmean)
        + k * torch.ger(secondlastmean, secondlastmean)
        ) / k
