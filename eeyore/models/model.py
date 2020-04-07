import hashlib
import torch
import torch.nn as nn

class Model(nn.Module):
    """ Class representing sampleable neural network model """
    def __init__(self, dtype=torch.float64, device='cpu'):
        super().__init__()
        self.dtype = dtype
        self.device = device

    def summary(self, hashsummary=False):
        print(self)
        print("-" * 80)
        n_params = self.num_params()
        print(f"Number of model parameters: {n_params}")
        print("-" * 80)

        if hashsummary:
            print('Hash Summary:')
            for idx, hashvalue in enumerate(self.hashsummary()):
                print(f"{idx}: {hashvalue}")

    def hashsummary(self):
        """ Print a model summary - checksums of each layer parameters """
        children = list(self.children())

        result = []
        for child in children:
            result.extend(hashlib.sha256(x.detach().cpu().numpy().tobytes()).hexdigest() for x in child.parameters())

        return result

    def num_params(self):
        """ Get the number of model parameters. """
        return sum(p.numel() for p in self.parameters())

    def get_params(self):
        return torch.cat([p.view(-1) for p in self.parameters()])

    def get_grad(self):
        return torch.cat([p.grad.view(-1) for p in self.parameters()])

    def set_params(self, theta, grad_val=None):
        """ Set model parameters with theta. """
        i = 0
        for p in self.parameters():
            j = p.numel()
            p.data = theta[i:i+j].view(p.size())
            if p.grad is not None:
                p.grad.detach_()
                p.grad.zero_()
            if grad_val is not None:
                p.grad = grad_val[i:i+j].view(p.size())
            i += j
