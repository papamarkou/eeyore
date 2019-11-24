import hashlib

import torch
import torch.nn as nn
# import torch.nn.functional as F

from torch.utils.data import DataLoader

from torch.autograd import grad

from eeyore.stats import binary_cross_entropy

class Model(nn.Module):
    """ Class representing sampleable neural network model """
    def __init__(self, dtype=torch.float64, device='cpu'):
        super().__init__()
        self.dtype = dtype
        self.device = device

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
    def __init__(self, loss=lambda x, y: binary_cross_entropy(x, y, reduction='sum'), constraint=None,
    bounds=[None, None], temperature=None, dtype=torch.float64, device='cpu'):
    # Use the built-in binarry cross entropy 'F.binary_cross_entropy' once the relevant PyTorch issue is resolved
    # https://github.com/pytorch/pytorch/issues/18945
    # def __init__(self, loss=lambda x, y: F.binary_cross_entropy(x, y, reduction='sum'), temperature=None,
    # dtype=torch.float64, device='cpu'):
        super().__init__(dtype=dtype, device=device)
        self.loss = loss
        self.constraint = constraint
        self.bounds = bounds
        self.temperature = temperature

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

    def transform_params_forwards(self, theta):
        if (self.bounds[0] != -float('inf')) and (self.bounds[1] == float('inf')):
            theta_transformed = torch.log(theta - self.bounds[0])
        elif (self.bounds[0] == -float('inf')) and (self.bounds[1] != float('inf')):
            theta_transformed = torch.log(self.bounds[1] - theta)
        elif (self.bounds[0] != -float('inf')) and (self.bounds[1] != float('inf')):
            theta_transformed = - torch.log((self.bounds[1] - theta) / (theta - self.bounds[0]))

    def transform_params_backwards(self, theta):
        if (self.bounds[0] != -float('inf')) and (self.bounds[1] == float('inf')):
            theta_transformed = torch.exp(theta) + self.bounds[0]
        elif (self.bounds[0] == -float('inf')) and (self.bounds[1] != float('inf')):
            theta_transformed = self.bounds[1] - torch.exp(theta)
        elif (self.bounds[0] != -float('inf')) and (self.bounds[1] != float('inf')):
            theta_transformed = (self.bounds[1] - self.bounds[0]) / (1 + torch.exp(-theta)) + self.bounds[0]

    def transform_params(self, theta):
        if self.constraint == 'transformation':
            theta_transformed = self.transform_params_backwards(theta)
        elif self.constraint == 'truncation':
            theta_transformed = self.transform_params_forwards(theta)

        self.set_params(theta_transformed.clone().detach())

    def log_lik(self, x, y):
        """ Log-likelihood """
        log_lik_val = -self.loss(self(x), y)
        if self.temperature is not None:
            log_lik_val = self.temperature * log_lik_val
        return log_lik_val

    def log_prior(self):
        log_prior_val = torch.sum(self.prior.log_prob(self.get_params()))
        if self.temperature is not None:
            log_prior_val = self.temperature * log_prior_val
        return log_prior_val

    def log_target(self, theta, dataloader):
        self.set_params(theta)

        log_prior_val = self.log_prior()

        if self.constraint is not None:
            self.trunc_transform_params(theta)

        x, y, _ = next(iter(dataloader))

        log_lik_val = self.log_lik(x, y)

        if self.constraint is not None:
            self.set_params(theta)

        return log_lik_val + log_prior_val

    def grad_log_target(self, log_target_val):
        grad_log_target_val = grad(log_target_val, self.parameters(), create_graph=True)
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
            deriv_i_wrt_grad = grad(grad_log_target_val[i], self.parameters(), retain_graph=True)
            hess_log_target_val.append(torch.cat([h.view(-1) for h in deriv_i_wrt_grad]))
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

    def predictive_posterior(self, x, y, chain): # (x, y) must be a single data point
        n_thetas = len(chain)
        predictive_posterior_val = 0

        for i in range(n_thetas):
            if self.constraint is not None:
                self.transform_params(chain.vals['theta'][i].clone().detach())
            else:
                self.set_params(chain.vals['theta'][i].clone().detach())

            predictive_posterior_val = predictive_posterior_val + torch.exp(self.log_lik(x, y))

        predictive_posterior_val = predictive_posterior_val / n_thetas

        return predictive_posterior_val

    def predictive_posterior_sample(self, num_samples, dataset, chain, shuffle=True):
        dataloader = DataLoader(dataset, batch_size=1, shuffle=shuffle)
        dataloader_iterator = iter(dataloader)
        samples = torch.empty(num_samples, dtype=self.dtype, device=self.device)
        indices = torch.empty(num_samples, dtype=self.dtype, device=self.device)

        for i in range(num_samples):
            x, y, idx = next(dataloader_iterator)
            samples[i] = self.predictive_posterior(x, y, chain)

        return samples, indices
