import torch

from torch.autograd import grad

class Density:
    def __init__(self, log_target, theta=None, constraint=None, bounds=[None, None], temperature=None,
    dtype=torch.float64, device='cpu'):
        self.theta = theta
        self.constraint = constraint
        self.bounds = bounds
        self.temperature = temperature
        self.dtype = dtype
        self.device = device

        self._log_target = log_target

    def get_grad(self):
        return self.theta.grad

    def set_params(self, theta):
        self.theta = theta

    def num_params(self):
        return len(self.theta)

    def summary(self, hashsummary=False):
        print(self)
        print("-" * 80)
        n_params = self.num_params()
        print(f"Number of density parameters: {n_params}")
        print("-" * 80)

    def log_target(self, theta, dataloader):
        self.set_params(theta.clone().detach())
        self.theta.requires_grad_(True)
        result = self._log_target(self.theta, dataloader)
        if self.temperature is not None:
            result = self.temperature * result
        return result

    def grad_log_target(self, log_target_val):
        grad_log_target_val = grad(log_target_val, self.theta, create_graph=True)
        grad_log_target_val = torch.cat([g.view(-1) for g in grad_log_target_val])
        return grad_log_target_val

    def upto_grad_log_target(self, theta, dataloader):
        log_target_val = self.log_target(theta, dataloader)
        grad_log_target_val = self.grad_log_target(log_target_val)
        return log_target_val, grad_log_target_val

    def hess_log_target(self, grad_log_target_val):
        n_params = self.num_params()

        hess_log_target_val = []
        for i in range(n_params):
            hess_log_target_val.append(grad(grad_log_target_val[i], self.theta, retain_graph=True)[0])
        hess_log_target_val = torch.cat(hess_log_target_val, 0).reshape(n_params, n_params)

        return hess_log_target_val

    def metric_log_target(self, grad_log_target_val):
        return -self.hess_log_target(grad_log_target_val)

    def upto_hess_log_target(self, theta, dataloader):
        log_target_val, grad_log_target_val = self.upto_grad_log_target(theta, dataloader)
        hess_log_target_val = self.hess_log_target(grad_log_target_val)
        return log_target_val, grad_log_target_val, hess_log_target_val

    def upto_metric_log_target(self, theta, dataloader):
        log_target_val, grad_log_target_val, hess_log_target_val = self.upto_hess_log_target(theta, dataloader)
        return log_target_val, grad_log_target_val, -hess_log_target_val
