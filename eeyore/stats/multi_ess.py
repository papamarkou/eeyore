import torch

from .cov import cov
from .mc_cov import mc_cov

def multi_ess(x, method='inse', adjust=False):
    num_iters, num_pars = x.shape

    cov_mat_det = torch.det(cov(x, rowvar=False)).item()
    mc_cov_mat_det = torch.det(mc_cov(x, method=method, adjust=adjust)).item()

    return num_iters * ((cov_mat_det / mc_cov_mat_det) ** (1/num_pars))
