import torch

def mc_se_from_cov(x):
    return torch.diag(x).sqrt()
