# https://www.sciencedirect.com/science/article/pii/S0047259X16301877
# https://arxiv.org/pdf/1706.00853.pdf
# The notation follows the implementation of insec.cpp of the mcmcse R package

import torch

from eeyore.linalg import is_pos_def

def inse_mc_cov(x, adjust=False):
    x_ctr = x - x.mean(0)

    n, p = x.shape

    ub = torch.floor(torch.tensor(n / 2, dtype=x.dtype)).int().item()
    sn = ub

    if adjust:
        Gamadj = torch.zeros([p, p], dtype=x.dtype)

    for m in range(ub):
        gam0 = torch.zeros([p, p], dtype=x.dtype)
        gam1 = torch.zeros([p, p], dtype=x.dtype)

        for i in range(n - 2 * m):
            gam0 = gam0 + torch.ger(x_ctr[i, :], x_ctr[i + 2 * m, :])
        gam0 = gam0 / n

        for i in range(n - 2 * m - 1):
            gam1 = gam1 + torch.ger(x_ctr[i, :], x_ctr[i + 2 * m + 1, :])
        gam1 = gam1 / n

        Gam = gam0 + gam1
        Gam = (Gam + Gam.t()) / 2

        if m == 0:
            Sig = -gam0 + 2 * Gam
        else:
            Sig = Sig + 2 * Gam

        if is_pos_def(Sig):
            sn = m
            break

    if sn > (ub - 1):
        raise RuntimeError('Not enough samples')

    last_dtm = torch.det(Sig).item()

    for m in range(sn + 1, ub):
        gam0 = torch.zeros([p, p], dtype=x.dtype)
        gam1 = torch.zeros([p, p], dtype=x.dtype)

        for i in range(n - 2 * m):
            gam0 = gam0 + torch.ger(x_ctr[i, :], x_ctr[i + 2 * m, :])
        gam0 = gam0 / n

        for i in range(n - 2 * m - 1):
            gam1 = gam1 + torch.ger(x_ctr[i, :], x_ctr[i + 2 * m + 1, :])
        gam1 = gam1 / n

        Gam = gam0 + gam1
        Gam = (Gam + Gam.t()) / 2

        Sig1 = Sig + 2 * Gam

        current_dtm = torch.det(Sig1).item()

        if current_dtm <= last_dtm:
            break

        Sig = Sig1.clone().detach()

        last_dtm = current_dtm

        if adjust:
            eigenvals, eigenvecs = torch.symeig(Gam, eigenvectors=True)
            eigenvals[eigenvals > 0] = 0
            Gamadj = Gamadj - eigenvecs @ torch.diag(eigenvals) @ eigenvecs.t()

    if adjust:
        Sig = Sig + 2 * Gamadj

    return Sig
