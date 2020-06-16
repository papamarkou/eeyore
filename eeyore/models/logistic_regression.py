import torch
import torch.nn as nn

from torch.distributions import Normal

from .bayesian_model import BayesianModel

class Hyperparameters:
    def __init__(self, input_size=1, output_size=1, bias=True, activation=torch.sigmoid):
        self.input_size = input_size
        self.output_size = output_size
        self.bias = bias
        self.activation = activation

class LogisticRegression(BayesianModel):
    def __init__(self, loss, constraint=None, bounds=[None, None], temperature=None, prior=None,
    hparams=Hyperparameters(), savefile=None, dtype=torch.float64, device='cpu'):
        super().__init__(
            loss=loss, constraint=constraint, bounds=bounds, temperature=temperature, dtype=dtype, device=device)
        self.hp = hparams
        self.linear = nn.Linear(self.hp.input_size, self.hp.output_size, bias=self.hp.bias).to(
            dtype=self.dtype, device=self.device
        )
        self.prior = prior or self.default_prior()
        if savefile:
            self.load_state_dict(savefile, strict=False)

    def default_prior(self):
        return Normal(
            torch.zeros(self.num_params(), dtype=self.dtype, device=self.device),
            torch.ones(self.num_params(), dtype=self.dtype, device=self.device)
        )

    def forward(self, x):
        x = self.linear(x)
        if self.hp.activation is not None:
            x = self.hp.activation(x)
        return x
