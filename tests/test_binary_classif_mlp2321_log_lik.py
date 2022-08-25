# %% Import packages

import torch
import unittest

from eeyore.constants import loss_functions
from eeyore.datasets import XYDataset
from eeyore.models.mlp import Hyperparameters, MLP

# %% Load XOR data

xor = XYDataset.from_eeyore('xor', dtype=torch.float64)

data = xor.x
labels = xor.y

# %% Set model parameters

theta = torch.tensor(
    [1.1, -2.9, -0.4, 0.8, 4.3, 9.2, 4.44, -3.4, 7.2, 1.2, -2.3, 0.4, -5.4, -3.3, 2.8, 2.9, 7.7, -4.4, 2,6],
    dtype=torch.float64
)

# %% Set MLP model

hparams = Hyperparameters(dims=[2, 3, 2, 1], bias=3*[True], activations=3*[torch.sigmoid])

model = MLP(
    loss=loss_functions['binary_classification'],
    hparams=hparams,
    dtype=torch.float64
)

# %% Compute MLP log-likelihood using eeyore API version

model.set_params(theta.clone().detach())

result01 = model.log_lik(data, labels)

# print("Log-likelihood based on MLP model:", result01.item())

# %% Define log_lik function

def log_lik(g, y):
    term = y * torch.log(torch.sigmoid(g)) + (1-y) * torch.log(1-torch.sigmoid(g))
    return torch.sum(term)

# %% Compute MLP log-likelihood manually following hidden layer node structure

def forward02(x):
    g11 = x @ theta[[0, 1]] + theta[6]
    g12 = x @ theta[[2, 3]] + theta[7]
    g13 = x @ theta[[4, 5]] + theta[8]
    h11 = torch.sigmoid(g11)
    h12 = torch.sigmoid(g12)
    h13 = torch.sigmoid(g13)
    g21 = torch.stack([h11, h12, h13]).t() @ theta[[9, 10, 11]] + theta[15]
    g22 = torch.stack([h11, h12, h13]).t() @ theta[[12, 13, 14]] + theta[16]
    h21 = torch.sigmoid(g21)
    h22 = torch.sigmoid(g22)
    g31 = torch.stack([h21, h22]).t() @ theta[[17, 18]] + theta[19]
    h31 = torch.sigmoid(g31)

    return h31[:, None]

out_of_forward02 = forward02(data)

result02 = log_lik(torch.logit(out_of_forward02), labels)

# print("Log-likelihood based on manual evaluation following hidden layer node structure:", result02.item())

# %% Class for running tests

class TestLogLiks(unittest.TestCase):
    def test_result01_vs_result02(self):
        self.assertEqual(result01.item(), result02.item())

# %% Enable running the tests from the command line

if __name__ == '__main__':
    unittest.main()
