import torch

def softabs(hessian, a=1000.0):
    l, Q = torch.linalg.eigh(hessian, "U")
    return Q @ torch.diag(torch.div(l, torch.tanh(a * l))) @ Q.t()
