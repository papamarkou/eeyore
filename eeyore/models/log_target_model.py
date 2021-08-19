import torch

from torch.autograd import grad

from .model import Model

class LogTargetModel(Model):
    def __init__(self, temperature=None, dtype=torch.float64, device='cpu'):
        super().__init__(dtype=dtype, device=device)
        self.temperature = temperature

    def log_target(self, theta, x, y):
        raise NotImplementedError

    def grad_log_target(self, log_target_val):
        grad_log_target_val = grad(log_target_val, self.parameters(), create_graph=True)
        grad_log_target_val = torch.cat([g.view(-1) for g in grad_log_target_val])
        return grad_log_target_val

    def upto_grad_log_target(self, theta, x, y):
        log_target_val = self.log_target(theta, x, y)
        grad_log_target_val = self.grad_log_target(log_target_val)
        return log_target_val, grad_log_target_val
