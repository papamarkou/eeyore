# %% Evaluation of grad of MLP log-target for binary classification
# 
# Confirm PyTorch and manually coded grad and Hessian of MLP log-target coincide

# %% Import packages

import torch
import unittest

from torch.autograd import grad
from torch.distributions import Normal
from torch.utils.data import DataLoader

from eeyore.constants import loss_functions
from eeyore.datasets import XYDataset
from eeyore.models.mlp import Hyperparameters, MLP
from eeyore.stats import binary_cross_entropy

# %% Compute MLP log-target using eeyore API version

# Load XOR data

xor = XYDataset.from_eeyore('xor', dtype=torch.float64)

data = xor.x
labels = xor.y

dataloader = DataLoader(xor, batch_size=4, shuffle=False)

# Setup MLP model

hparams = Hyperparameters([2, 2, 1])
model = MLP(
    loss=loss_functions['binary_classification'],
    hparams=hparams,
    dtype=torch.float64
)
model.prior = Normal(torch.zeros(9, dtype=torch.float64), 100*torch.ones(9, dtype=torch.float64))

# Fix model parameters

theta0 = torch.tensor([1.1, -2.9, -0.4, 0.8, 4.3, 9.2, 4.44, -3.4, 7.2], dtype=torch.float64)
theta = theta0.clone().detach()
model.set_params(theta.clone().detach())

# Compute MLP log-target using eeyore API version

lt_result01 = model.log_target(theta, data, labels)

# Compute sum of log-lik and log-prior using eeyore API version

log_lik_plus_log_prior = model.log_lik(data, labels)+model.log_prior()

# %% Compute MLP log-target fully manually

# Define function for computing log-likelihood manually

def log_lik(theta, x, y):
    w1 = theta[0:4].view(2, 2)
    b1 = theta[4:6].view(2)
    g1 = x @ w1.t() + b1
    h1 = torch.sigmoid(g1)
    w2 = theta[6:8].view(1, 2)
    b2 = theta[8:9].view(1)
    g2 = h1 @ w2.t() + b2
    h2 = torch.sigmoid(g2)
    
    return -binary_cross_entropy(h2, y, reduction='sum')

# Define function for computing log-prior manually

def log_prior(theta):
    d = Normal(torch.zeros(9, dtype=torch.float64), 100*torch.ones(9, dtype=torch.float64))
    return torch.sum(d.log_prob(theta))

# Define function for computing log-target manually

def log_target(theta, x, y):
    return log_lik(theta, x, y)+log_prior(theta)

# Compute log-target manually

lt_result02 = log_target(theta, data, labels)

# %% Compute grad of MLP log-target using eeyore API version

theta = theta0.clone().detach()

lt_val01 = model.log_target(theta, data, labels)

glt_result01 = model.grad_log_target(lt_val01)

# %% Compute grad of MLP log-target using backward pass

theta = theta0.clone().detach()

lt_val02 = model.log_target(theta, data, labels)

lt_val02.backward(retain_graph=True)

# Rerun so that it becomes possible to call p.grad.zero_()

theta = theta0.clone().detach()

lt_val03 = model.log_target(theta, data, labels)

for p in model.parameters():
    p.grad.zero_()

lt_val03.backward()

glt_result02 = torch.cat([p.grad.view(-1) for p in model.parameters()])

# %% Compute grad of MLP log-target calling grad() on manually coded log_target()

theta = theta0.clone().detach()
theta.requires_grad_(True)

lt_val04 = log_target(theta, data, labels)

glt_result03, = grad(lt_val04, theta)

# %% Compute grad of MLP log-target calling grad() on manually coded log-lik and log-prior

theta = theta0.clone().detach()
theta.requires_grad_(True)

ll_val = log_lik(theta, data, labels)

gll_val, = grad(ll_val, theta)

lp_val = log_prior(theta)

glp_val, = grad(lp_val, theta)

glt_result04 = gll_val+glp_val

# %% Class for running target tests

class TestTargets(unittest.TestCase):
    def test_lt_result01_vs_log_lik_plus_log_prior(self):
        self.assertEqual(lt_result01.item(), log_lik_plus_log_prior.item())
        
    def test_lt_result01_vs_lt_result02(self):
        self.assertEqual(lt_result01.item(), lt_result02.item())

# %% Class for running gradient tests

class TestGradients(unittest.TestCase):        
    def test_glt_result01_vs_glt_result02(self):
        self.assertTrue(torch.equal(glt_result01, glt_result02))

    def test_glt_result01_vs_glt_result03(self):
        self.assertTrue(torch.equal(glt_result01, glt_result03))

    def test_glt_result01_vs_glt_result04(self):
        self.assertTrue(torch.equal(glt_result01, glt_result04))

# %% Enable running the tests from the command line

if __name__ == '__main__':
    unittest.main()
