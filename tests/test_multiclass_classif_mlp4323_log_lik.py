# %% Import packages

import torch
import unittest

from eeyore.constants import loss_functions
from eeyore.datasets import XYDataset
from eeyore.models.mlp import Hyperparameters, MLP

# %% Load iris data

iris = XYDataset.from_eeyore('iris', yndmin=1, yonehot=True, dtype=torch.float64)

data = iris.x
labels = iris.y

# %% Set model parameters

theta = torch.tensor([
    0.2213, 0.5852, 0.1458, 0.5139, -0.1946, 0.0489, -0.1281, -0.7307,
    0.2176, 0.3274, -1.3060, 0.3253, -0.4248, 1.7403, 0.6219, 0.2652,
    -0.5310, -0.0291, 1.0262, -0.4920, 0.4391, -0.2450, 2.3145, -0.0788,
    1.1180, -1.2803, -0.4435, 0.5371, -0.2440, -0.3574, 0.4446, -0.3453],
    dtype=torch.float64
)

# %% Set MLP model

hparams = Hyperparameters(dims=[4, 3, 2, 3], bias=[True, True, True], activations=[torch.sigmoid, torch.sigmoid, None])

model = MLP(
    loss=loss_functions['multiclass_classification'],
    hparams=hparams,
    dtype=torch.float64
)

# %% Compute MLP log-likelihood using eeyore API version

model.set_params(theta.clone().detach())

result01 = model.log_lik(data, labels)

# print("Log-likelihood based on MLP model:", result01.item())

# %% Define log_lik function

def log_lik(g, y):
    labels_argmax = torch.argmax(labels, 1)
    g_log = torch.log(g)
    result = sum([g_log[i, labels_argmax[i]] for i in range(len(labels_argmax))])

    return result

# %% Compute MLP log-likelihood manually following hidden layer node structure

def forward02(x):
    g11 = x @ theta[[0, 1, 2, 3]] + theta[12]
    g12 = x @ theta[[4, 5, 6, 7]] + theta[13]
    g13 = x @ theta[[8, 9, 10, 11]] + theta[14]
    h11 = torch.sigmoid(g11)
    h12 = torch.sigmoid(g12)
    h13 = torch.sigmoid(g13)
    g21 = torch.stack([h11, h12, h13]).t() @ theta[[15, 16, 17]] + theta[21]
    g22 = torch.stack([h11, h12, h13]).t() @ theta[[18, 19, 20]] + theta[22]
    h21 = torch.sigmoid(g21)
    h22 = torch.sigmoid(g22)
    g31 = torch.stack([h21, h22]).t() @ theta[[23, 24]] + theta[29]
    g32 = torch.stack([h21, h22]).t() @ theta[[25, 26]] + theta[30]
    g33 = torch.stack([h21, h22]).t() @ theta[[27, 28]] + theta[31]

    result = torch.vstack([g31, g32, g33])

    return result

out_of_forward02 = forward02(data)

probabilities = torch.exp(out_of_forward02)
probabilities = probabilities / probabilities.sum(axis=0)
probabilities = probabilities.t()

result02 = log_lik(probabilities, labels)

# print("Log-likelihood based on manual evaluation following hidden layer node structure:", result02.item())

# %% Class for running tests

class TestLogLiks(unittest.TestCase):
    def test_result01_vs_result02(self):
        self.assertAlmostEqual(result01.item(), result02.item(), places=12)

# %% Enable running the tests from the command line

if __name__ == '__main__':
    unittest.main()
