import torch
import torch.nn as nn
from torch.distributions import Normal

from eeyore.api import BayesianModel

class Hyperparameters:
    def __init__(self, dims=[1, 2, 1], activations=2*[torch.sigmoid]):
        self.dims = dims
        self.activations = activations

        if len(self.dims) < 3:
            raise ValueError

        if (len(self.dims) != len(self.activations)+1):
            raise ValueError

class MLP(BayesianModel):
    def __init__(self, hparams=Hyperparameters(), savefile=None, dtype=torch.float64):
        super().__init__(dtype=dtype)
        self.hp = hparams
        self.fc_layers = self.set_fc_layers()
        self.prior = self.default_prior()
        if savefile:
            self.load_state_dict(savefile, strict=False)

    def default_prior(self):
        distro = Normal(
            torch.zeros(self.num_params(), dtype=self.dtype),
            torch.ones(self.num_params(), dtype=self.dtype)
        )
        return distro

    def set_fc_layers(self):
        fc = []
        for i in range(len(self.hp.dims)-1):
            fc.append(nn.Linear(self.hp.dims[i], self.hp.dims[i+1]).to(dtype=self.dtype))
        return nn.ModuleList(fc)

    def forward(self, x):
        for fc, activation in zip(self.fc_layers, self.hp.activations):
            x = activation(fc(x))
        return x

    def num_hidden_layers(self):
        """ Get the number of hidden layers. """
        return len(self.hp.dims)-2
