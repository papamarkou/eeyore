import math
import torch
import torch.nn as nn

from torch.distributions import Normal, MultivariateNormal
from torchdiffeq import odeint

from eeyore.models import BayesianModel

class ODEModel(BayesianModel):
    def __init__(self, odes, eta, z0, noise_var=None, known_noise_var=False, constraint=None, bounds=[None, None],
        temperature=None, prior=None, savefile=None, dtype=torch.float64, device='cpu'):
        super().__init__(
            loss=None, constraint=constraint, bounds=bounds, temperature=temperature, dtype=dtype, device=device
        )

        self.z0 = z0

        # eta are the ODE parameters and n_v is the noise variance
        self.eta = nn.Parameter(data=eta.to(dtype=self.dtype, device=self.device), requires_grad=True)

        if noise_var is None:
            self.n_v = torch.empty(self.num_states(), dtype=self.dtype, device=self.device)
        else:
            self.n_v = noise_var.to(dtype=self.dtype, device=self.device)

        if not known_noise_var:
            self.n_v = nn.Parameter(data=self.n_v, requires_grad=True)

        self.loss = self.columnwise_isotropic_normal_loss

        self.odes = odes

        self.prior = prior or self.default_prior()

    def columnwise_isotropic_normal_loss(self, z, y):
        return 0.5 * (
            torch.sum((y - z).pow(2).sum(0) / self.n_v)
            + self.num_times(z) * torch.sum(torch.log(self.n_v))
            + self.num_times(z) * self.num_states() * torch.log(torch.tensor(
                [2 * math.pi], dtype=self.dtype, device=self.device
            ))
        )

    def default_prior(self):
        return Normal(
            torch.zeros(self.num_params(), dtype=self.dtype, device=self.device),
            torch.ones(self.num_params(), dtype=self.dtype, device=self.device)
        )

    def num_states(self):
        return len(self.z0)

    def num_times(self, t):
        return t.shape[0]

    def forward(self, t):
        return odeint(lambda t, z: self.odes(t, z, self.eta), self.z0, t)

    def sample(self, t):
        z = self(t)

        y = torch.empty(self.num_times(t), self.num_states())
        for i in range(self.num_states()):
            y[:, i] = MultivariateNormal(
                    loc=z[:, i],
                    covariance_matrix=torch.diag(torch.tensor(
                        [self.n_v[i] for _ in range(self.num_times(t))], dtype=self.dtype, device=self.device)
                    )
                ).sample()

        return y, z
