# See lemma 2 of section 4.1 in https://www.tandfonline.com/doi/abs/10.1080/10618600.1998.10474787

import torch

from .cov import cov
from .mc_cov import mc_cov

# x is a numpy array of 3 dimensions, (chain, MC iteration, parameter)
def multi_rhat(x, method='inse', adjust=False):
    num_chains, num_iters, num_pars = x.shape

    w = torch.zeros([num_pars, num_pars])
    for i in range(num_chains):
        w = w + mc_cov(x[i], method=method, adjust=adjust)
    w = w / num_chains

    b = cov(x.mean(1), rowvar=False)

    rhat = max(torch.symeig(torch.matmul(torch.inverse(w), b))[0]).item()
    rhat = ((num_iters - 1) / num_iters) + ((num_chains + 1) / num_chains) * rhat

    return rhat, w, b
