# %% Evaluation of MLP log-likelihood for multiclass classification
# 
# Confirm PyTorch and manually coded MLP log-likelihood coincide

# %% Import packages

import torch
import torch.nn as nn

from eeyore.constants import loss_functions
from eeyore.datasets import XYDataset
from eeyore.models.mlp import Hyperparameters, MLP

# %% Compute MLP log-likelihood using eeyore API version

# Load iris data

iris = XYDataset.from_eeyore('iris', yndmin=1, yonehot=True, dtype=torch.float64)

data = iris.x
labels = iris.y

# Setup MLP model

hparams = Hyperparameters(dims=[4, 3, 3], activations=[torch.sigmoid, None])
model = MLP(
    loss=loss_functions['multiclass_classification'],
    hparams=hparams,
    dtype=torch.float64
)

# Fix model parameters

# theta = torch.rand(27, dtype=torch.float64)
theta = torch.tensor([
    0.7735, 0.8161, 0.3910, 0.9622, 0.3748, 0.8711, 0.3315, 0.5473, 0.8820,
    0.0294, 0.9686, 0.8313, 0.6693, 0.8791, 0.6271, 0.8636, 0.3814, 0.0319,
    0.5148, 0.5086, 0.7428, 0.5464, 0.5278, 0.6127, 0.4499, 0.1538, 0.9291], dtype=torch.float64)
model.set_params(theta.clone().detach())

# Compute MLP log-likelihood using eeyore API version

result01 = model.log_lik(data, labels)

# %% Compute MLP log-likelihood using eeyore loss method in MLP model

result02 = -model.loss(model(data), labels)

# model(data).softmax(1);
# torch.argmax(model(data).softmax(1), 1)

# %% Compute MLP log-likelihood using Pytorch cross entropy

result03 = -nn.CrossEntropyLoss(reduction='sum')(model(data), torch.argmax(labels, 1))

# %% Compute MLP log-likelihood using Pytorch negative log-likelihood loss

result04 = -nn.NLLLoss(reduction='sum')(nn.Softmax(dim=1)(model(data)).log(), torch.argmax(labels, 1))

# %% Compute MLP log-likelihood manually

def cross_entropy_loss(data, labels):
    n = labels.size(dim=0)
    
    logit = model(data)
    
    softmax_vals = nn.Softmax(dim=1)(logit).log()
    labels_argmax = torch.argmax(labels, 1)
    
    result = 0
    for i in range(n):
        result = result + softmax_vals[i, labels_argmax[i]]
    result = result
        
    return result

result05 = cross_entropy_loss(data, labels)

# %% Run tests

class TestLogLiks:
    def test_result01_vs_result02(self):
        assert torch.equal(result01, result02)

    def test_result01_vs_result03(self):
        assert torch.equal(result01, result03)

    def test_result01_vs_result04(self):
        assert torch.equal(result01, result04)

    def test_result01_vs_result05(self):
        assert torch.equal(result01, result05)
