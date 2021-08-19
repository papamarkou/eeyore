import torch

def cor_from_cov(x):
    num_pars = x.shape[1]
    sigma_inv = (1 / torch.diag(x).sqrt()).expand(num_pars, num_pars)

    return x * sigma_inv * sigma_inv.t()
