import torch

def is_pos_def(x):
    if torch.equal(x, torch.t(x)):
        try:
            torch.cholesky(x)
            return True
        except RuntimeError:
            return False
    else:
        return False
