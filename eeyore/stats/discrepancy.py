import torch

def squared_mmd(x1, x2, kernel, biased=True):
    n1 = len(x1)
    n2 = len(x2)

    if biased:
        return (kernel.sum_symm_K(x1, include_diag=True) / (n1 ** 2)
            + kernel.sum_symm_K(x2, include_diag=True) / (n2 ** 2)
            - 2 * kernel.sum_K(x1, x2) / (n1 * n2)
        )
    else:
        return (kernel.sum_symm_K(x1, include_diag=False) / (n1 * (n1 - 1))
            + kernel.sum_symm_K(x2, include_diag=False) / (n2 * (n2 - 1))
            - 2 * kernel.sum_K(x1, x2) / (n1 * n2)
        )

def mmd(x1, x2, kernel):
    return torch.sqrt(squared_mmd(x1, x2, kernel, biased=True))
