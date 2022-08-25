import itertools
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
    def __init__(self, loss, temperature=None, prior=None, hparams=Hyperparameters(), savefile=None, dtype=torch.float64,
    device='cpu'):
        super().__init__(loss, temperature=temperature, dtype=dtype, device=device)
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

    def num_par_blocks(self):
        return sum(self.hp.dims[1:])

    def layer_and_node_from_par_block(self, b):
        num_nodes_per_layer = [0] + list(itertools.accumulate(self.hp.dims[1:]))
        l = self.num_hidden_layers()

        for i in range(1, len(num_nodes_per_layer)):
            if num_nodes_per_layer[-i-1] <= b < num_nodes_per_layer[-i]:
                n = b if (num_nodes_per_layer[-i-1] == 0) else (b % num_nodes_per_layer[-i-1])
                break
            else:
                l = l - 1

        return l, n

    def starting_par_block_idx(self, l):
        s = 0

        if l > 0:
            for i in range(l):
                s = s + (self.hp.dims[i]+1 if self.hp.bias[i] else self.hp.dims[i])*self.hp.dims[i+1]

        return s

    def starting_par_block_indices(self):
        s = [0]

        for l in range(self.num_hidden_layers()):
            s.append(s[-1]+(self.hp.dims[l]+1 if self.hp.bias[l] else self.hp.dims[l])*self.hp.dims[l+1])

        return s

    def par_block_indices(self, b):
        l, n = self.layer_and_node_from_par_block(b)
        s = self.starting_par_block_idx(l)

        indices = list(range(s+n*self.hp.dims[l], s+(n+1)*self.hp.dims[l])) if (self.hp.dims[l] > 1) else [s+n]

        if self.hp.bias[l]:
            indices.append(s+self.hp.dims[l]*self.hp.dims[l+1]+n)

        return indices, l, n
