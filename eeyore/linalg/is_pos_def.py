import torch

def is_pos_def(x):
    if torch.equal(x, x.t()):
        try:
            torch.linalg.cholesky(x)
            return True
        except RuntimeError:
            return False
    else:
        return False
