# See lemma 2 of section 4.1 in https://www.tandfonline.com/doi/abs/10.1080/10618600.1998.10474787

import torch

from eeyore.linalg import is_pos_def, nearest_pd

from .cov import cov
from .mc_cov import mc_cov

def multi_rhat(x, mc_cov_mat=None, method='inse', adjust=False):
    num_chains, num_iters, num_pars = x.shape

    w = torch.zeros([num_pars, num_pars])
    for i in range(num_chains):
        if mc_cov_mat is None:
            w = w + mc_cov(x[i], method=method, adjust=adjust, rowvar=False)
        else:
            w = w + mc_cov_mat[i]
    w = w / num_chains

    if not is_pos_def(w):
        w = nearest_pd(w)
        is_w_pd = False
    else:
        is_w_pd = True

    b = cov(x.mean(1), rowvar=False)

    if not is_pos_def(b):
        b = nearest_pd(b)
        is_b_pd = False
    else:
        is_b_pd = True

    eigvals = torch.linalg.eigvals(torch.matmul(torch.inverse(w), b))
    eigvals_argmax = eigvals.real.argmax().item()
    rhat = eigvals.real[eigvals_argmax].item()
    rhat = ((num_iters - 1) / num_iters) + ((num_chains + 1) / num_chains) * rhat

    return rhat, eigvals.imag[eigvals_argmax].item(), w, b, is_w_pd, is_b_pd
