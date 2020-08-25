import torch

from .mc_cov import mc_cov

def mc_se(x, cov_matrix=None, method='inse', adjust=False, rowvar=False):
    return torch.diag(mc_cov(x, method=method, adjust=adjust, rowvar=rowvar) if cov_matrix is None else cov_matrix).sqrt()
