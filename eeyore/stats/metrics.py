import torch

def softabs(hessian, a=1000.0):
    l, Q = torch.symeig(hessian, eigenvectors=True, upper=True)
    return Q @ torch.diag(torch.div(l, torch.tanh(a * l))) @ Q.t()
