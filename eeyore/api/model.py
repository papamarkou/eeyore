import hashlib

import torch
import torch.nn as nn
# import torch.nn.functional as F

from torch.autograd import grad

from eeyore.stats import binary_cross_entropy

class Model(nn.Module):
    """ Class representing sampleable neural network model """
    def __init__(self, dtype=torch.float64):
        super().__init__()
        self.dtype = dtype

    def num_params(self):
        """ Get the number of model parameters. """
        return sum(p.numel() for p in self.parameters())

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

class BayesianModel(Model):
    """ Class representing a Bayesian Net """
    def __init__(self, loss=lambda x, y: binary_cross_entropy(x, y, reduction='sum'), dtype=torch.float64):
    # Use the built-in binarry cross entropy 'F.binary_cross_entropy' once the relevant PyTorch issue is resolved
    # https://github.com/pytorch/pytorch/issues/18945
    # def __init__(self, loss=lambda x, y: F.binary_cross_entropy(x, y, reduction='sum'), dtype=torch.float64):
        super().__init__(dtype=dtype)
        self.loss = loss

    def default_prior(self):
        """ Prior distribution """
        raise NotImplementedError

    def summary(self, hashsummary=False):
        print(self)
        print("-" * 80)
        n_params = self.num_params()
        print(f"Number of model parameters: {n_params}")
        print("-" * 80)
        print(f"Prior: {self.prior}")
        print("-" * 80)

        if hashsummary:
            print('Hash Summary:')
            for idx, hashvalue in enumerate(self.hashsummary()):
                print(f"{idx}: {hashvalue}")

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
            if grad_val is not None:
                p.grad = grad_val[i:i+j].view(p.size())
            i += j

    def log_lik(self, x, y):
        """ Log-likelihood """
        return -self.loss(self(x), y)

    def log_prior(self):
        return torch.sum(self.prior.log_prob(self.get_params()))

    def log_target(self, theta, x, y):
        self.set_params(theta)
        return self.log_lik(x, y) + self.log_prior()

    def grad_log_target(self, log_target_val):
        grad_log_target_val = grad(log_target_val, self.parameters(), create_graph=True)
        grad_log_target_val = torch.cat([g.view(-1) for g in grad_log_target_val])
        return grad_log_target_val

    def upto_grad_log_target(self, theta, x, y):
        log_target_val = self.log_target(theta, x, y)
        grad_log_target_val = self.grad_log_target(log_target_val)
        return log_target_val, grad_log_target_val

    def hess_log_target(self, grad_log_target_val):
        n_params = self.num_params()

        hess_log_target_val = []
        for i in range(n_params):
            deriv_i_wrt_grad = grad(grad_log_target_val[i], self.parameters(), retain_graph=True)
            hess_log_target_val.append(torch.cat([h.view(-1) for h in deriv_i_wrt_grad]))
        hess_log_target_val = torch.cat(hess_log_target_val, 0).reshape(n_params, n_params)

        return hess_log_target_val

    def metric_log_target(self, grad_log_target_val):
        return -self.hess_log_target(grad_log_target_val)

    def upto_hess_log_target(self, theta, x, y):
        log_target_val, grad_log_target_val = self.upto_grad_log_target(theta, x, y)
        hess_log_target_val = self.hess_log_target(grad_log_target_val)

        return log_target_val, grad_log_target_val, hess_log_target_val

    def upto_metric_log_target(self, theta, x, y):
        log_target_val, grad_log_target_val, hess_log_target_val = self.upto_hess_log_target(theta, x, y)
        return log_target_val, grad_log_target_val, -hess_log_target_val
