# Implementation taken from
# https://stackoverflow.com/questions/43238173/python-convert-matrix-to-positive-semi-definite/43244194#43244194

import numpy as np
import torch

from .is_pos_def import is_pos_def

def nearest_pd(A, f=np.spacing):
    """Find the nearest positive-definite matrix to input

    A Python/Numpy port of John D'Errico's `nearestSPD` MATLAB code [1], which credits [2]

    [1] https://www.mathworks.com/matlabcentral/fileexchange/42885-nearestspd

    [2] https://doi.org/10.1016/0024-3795(88)90223-6
    """

    B = (A + A.T) / 2
    _, s, V = torch.svd(B)

    # For a comparison with kanga, see the following:
    # https://github.com/pytorch/pytorch/issues/16076#issuecomment-477755364
    H = torch.matmul(V, torch.matmul(torch.diag(s), V.T))

    A2 = (B + H) / 2

    A3 = (A2 + A2.T) / 2

    if is_pos_def(A3):
        return A3

    spacing = f(torch.norm(A).item())
    I = torch.eye(A.shape[0])
    k = 1
    while not is_pos_def(A3):
        eigenvals = torch.eig(A3, eigenvectors=False)[0][:, 0]
        mineig = eigenvals.min().item()
        A3 += I * (-mineig * k**2 + spacing)
        k += 1

    return A3
