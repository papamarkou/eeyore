import torch
import torch.nn as nn

from torch.distributions import Normal

from .bayesian_model import BayesianModel

class Hyperparameters:
    def __init__(self, dims=[1, 2, 1], bias=2*[True], activations=2*[torch.sigmoid]):
        self.dims = dims
        self.bias = bias
        self.activations = activations

        if len(self.dims) < 3:
            raise ValueError

        if (len(self.dims) != len(self.activations)+1):
            raise ValueError

class MLP(BayesianModel):
    def __init__(self, loss, constraint=None, bounds=[None, None], temperature=None, prior=None,
    hparams=Hyperparameters(), savefile=None, dtype=torch.float64, device='cpu'):
        super().__init__(
            loss=loss, constraint=constraint, bounds=bounds, temperature=temperature, dtype=dtype, device=device)
        self.hp = hparams
        self.fc_layers = self.set_fc_layers()
        self.prior = prior or self.default_prior()
        if savefile:
            self.load_state_dict(savefile, strict=False)

    def default_prior(self):
        return Normal(
            torch.zeros(self.num_params(), dtype=self.dtype, device=self.device),
            torch.ones(self.num_params(), dtype=self.dtype, device=self.device)
        )

    def set_fc_layers(self):
        fc = []
        for i in range(len(self.hp.dims)-1):
            fc.append(nn.Linear(
                self.hp.dims[i], self.hp.dims[i+1], bias=self.hp.bias[i]
            ).to(dtype=self.dtype, device=self.device))
        return nn.ModuleList(fc)

    def forward(self, x):
        for fc, activation in zip(self.fc_layers, self.hp.activations):
            x = fc(x)
            if activation is not None:
                x = activation(x)
        return x

    def num_hidden_layers(self):
        """ Get the number of hidden layers. """
        return len(self.hp.dims)-2
