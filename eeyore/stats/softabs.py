import torch

def softabs(hessian, a=1000.0):
    l, Q = torch.eig(hessian, True)
    return Q @ torch.diag(torch.div(l[:, 0], torch.tanh(a*l[:, 0]))) @ Q.t()
