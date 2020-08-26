import torch

def running_mean(x, dim=0):
    if x.dim() == 1:
        return torch.cumsum(x, dim=0) / torch.arange(1, x.size(dim=0) + 1)
    elif x.dim() == 2:
        if dim == 0:
            return torch.cumsum(x, dim=0) / torch.arange(1, x.size(dim=0) + 1).view(-1, 1)
        elif dim == 1:
            return torch.cumsum(x, dim=1) / torch.arange(1, x.size(dim=1) + 1).view(1, -1)
