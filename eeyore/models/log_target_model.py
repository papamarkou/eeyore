import torch

from torch.autograd import grad

from .model import Model

class LogTargetModel(Model):
    def __init__(self, constraint=None, bounds=[None, None], temperature=None, dtype=torch.float64, device='cpu'):
        super().__init__(dtype=dtype, device=device)
        self.constraint = constraint
        self.bounds = bounds
        self.temperature = temperature

    def transform_params_forwards(self, theta):
        if (self.bounds[0] != -float('inf')) and (self.bounds[1] == float('inf')):
            theta_transformed = torch.log(theta - self.bounds[0])
        elif (self.bounds[0] == -float('inf')) and (self.bounds[1] != float('inf')):
            theta_transformed = torch.log(self.bounds[1] - theta)
        elif (self.bounds[0] != -float('inf')) and (self.bounds[1] != float('inf')):
            theta_transformed = - torch.log((self.bounds[1] - theta) / (theta - self.bounds[0]))
        return theta_transformed

    def transform_params_backwards(self, theta):
        if (self.bounds[0] != -float('inf')) and (self.bounds[1] == float('inf')):
            theta_transformed = torch.exp(theta) + self.bounds[0]
        elif (self.bounds[0] == -float('inf')) and (self.bounds[1] != float('inf')):
            theta_transformed = self.bounds[1] - torch.exp(theta)
        elif (self.bounds[0] != -float('inf')) and (self.bounds[1] != float('inf')):
            theta_transformed = (self.bounds[1] - self.bounds[0]) / (1 + torch.exp(-theta)) + self.bounds[0]
        return theta_transformed

    def transform_params(self, theta):
        if self.constraint == 'transformation':
            theta_transformed = self.transform_params_backwards(theta)
        elif self.constraint == 'truncation':
            theta_transformed = self.transform_params_forwards(theta)

        self.set_params(theta_transformed.clone().detach())

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

    def hess_log_target(self, grad_log_target_val):
        num_params = self.num_params()

        hess_log_target_val = []
        for i in range(num_params):
            deriv_i_wrt_grad = grad(grad_log_target_val[i], self.parameters(), retain_graph=True)
            hess_log_target_val.append(torch.cat([h.view(-1) for h in deriv_i_wrt_grad]))
        hess_log_target_val = torch.cat(hess_log_target_val, 0).reshape(num_params, num_params)

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
