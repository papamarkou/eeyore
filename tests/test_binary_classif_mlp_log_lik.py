# %% Evaluation of MLP log-likelihood for binary classification
# 
# Confirm PyTorch and manually coded MLP log-likelihood coincide

# %% Import packages

import torch
import torch.nn as nn
import torch.nn.functional as F

from eeyore.constants import loss_functions
from eeyore.datasets import XYDataset
from eeyore.models.mlp import Hyperparameters, MLP

# %% Compute MLP log-likelihood using eeyore API version

# Load XOR data

xor = XYDataset.from_eeyore('xor', dtype=torch.float64)

data = xor.x
labels = xor.y

# Setup MLP model

hparams = Hyperparameters([2, 2, 1])
model = MLP(
    loss=loss_functions['binary_classification'],
    hparams=hparams,
    dtype=torch.float64
)

# Fix model parameters

theta = torch.tensor([1.1, -2.9, -0.4, 0.8, 4.3, 9.2, 4.44, -3.4, 7.2], dtype=torch.float64)
model.set_params(theta.clone().detach())

# Compute MLP log-likelihood using eeyore API version

result01 = model.log_lik(data, labels)

# %% Compute MLP log-likelihood using Pytorch forward method, loss and cross entropy

out = model(data)
# out = model.forward(data)

# Define logit loss

criterion = nn.BCEWithLogitsLoss(reduction='sum')
loss = criterion(out, labels)

# Define logit function

def logit(p):
    return torch.log(p/(1-p))

# Compute MLP log-likelihood using Pytorch binary_cross_entropy_with_logits

result02a = -F.binary_cross_entropy_with_logits(logit(out), labels, reduction='sum')

# Compare result01 with result02a

def test_result01_vs_result02a():
    assert result01 == result02a

# Compute MLP log-likelihood using Pytorch binary_cross_entropy

result02b = -F.binary_cross_entropy(out, labels, reduction='sum')

# Compare result01 with result02b

def test_result01_vs_result02b():
    assert result01 == result02b

# %% Compute MLP log-likelihood manually using Pytorch forward output

# Define sigmoid function

def sigmoid(x):
    return 1 / (1 + torch.exp(-x))

# Define MLP log-lik

def log_lik(g, y):
    term = y * torch.log(sigmoid(g)) + (1-y) * torch.log(1-sigmoid(g))
    return torch.sum(term)

# Compute MLP log-lik

result03 = log_lik(logit(out), labels)

# Compare result01 with result03

def test_result01_vs_result03():
    assert result01 == result03

# %% Compute MLP log-likelihood similarly to model.log_lik

# Define MLP forward01

def forward01(x):
    h = x
    for fc, activation in zip(model.fc_layers, model.hp.activations):
        h = activation(fc(h))
    return h

# Compute MLP forward01

out_of_forward01 = forward01(data)

# Compute MLP log-lik given forward01 output

result04 = log_lik(logit(out_of_forward01), labels)

# Compare result01 with result04

def test_result01_vs_result04():
    assert result01 == result04

# %% Compute MLP log-likelihood manually by invoking linear layer and activation functions

# Define MLP forward02

def forward02(x, num_layers):
    h = x
    for i in range(num_layers):
        h = model.hp.activations[i](model.fc_layers[i](h))
    return h

# Compute MLP forward02

out_of_forward02 = forward02(data, 2)

# Compute MLP log-lik given forward02 output

result05 = log_lik(logit(out_of_forward02), labels)

# Compare result01 with result05

def test_result01_vs_result05():
    assert result01 == result05

# %% Compute MLP log-likelihood computing manually linear layers given activations

# Define MLP forward03

def forward03(x, num_layers):
    h = x
    for i in range(num_layers):
        h = model.hp.activations[i](h @ model.fc_layers[i].weight.t() + model.fc_layers[i].bias)
    return h

# Compute MLP forward03

out_of_forward03 = forward03(data, 2)

# Compute MLP log-lik given forward03 output

result06 = log_lik(logit(out_of_forward03), labels)

# Compare result01 with result06

def test_result01_vs_result06():
    assert result01 == result06

# %% Compute MLP log-likelihood fully manually

# Define MLP forward04

def forward04(x):
    w1 = theta[0:4].view(2, 2)
    b1 = theta[4:6].view(2)
    g1 = x @ w1.t() + b1
    h1 = torch.sigmoid(g1)
    w2 = theta[6:8].view(1, 2)
    b2 = theta[8:9].view(1)
    g2 = h1 @ w2.t() + b2
    h2 = torch.sigmoid(g2)
    
    return h2

# Compute MLP forward04

out_of_forward04 = forward04(data)

# Compute MLP log-lik given forward04 output

result07 = log_lik(logit(out_of_forward04), labels)

# Compare result01 with result07

def test_result01_vs_result07():
    assert result01 == result07
