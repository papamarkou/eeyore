import numpy as np
import torch

import eeyore.stats as st

from pathlib import Path

from kanga.chains import ChainArray

from .chain import Chain

class ChainList(Chain):
    """ Monte Carlo chain to store samples in lists """

    def __init__(self, keys=['sample', 'target_val', 'accepted'], vals=None):
        self.reset(keys=keys, vals=vals)

    def reset(self, keys=['sample', 'target_val', 'accepted'], vals=None):
        if vals is None:
            self.vals = {key : [] for key in keys}
        else:
            self.vals = vals

    def __repr__(self):
        return f"Markov chain containing {len(self.vals['sample'])} samples."

    def __len__(self):
        return len(self.get_samples())

    def num_params(self):
        return len(self.get_sample(0))

    def get_param(self, idx):
        return torch.stack([sample[idx] for sample in self.vals['sample']])

    def get_sample(self, idx):
        return self.vals['sample'][idx]

    def get_samples(self):
        return torch.stack(self.vals['sample'])

    def get_target_vals(self):
        return torch.stack(self.vals['target_val'])

    def get_grad_val(self, idx):
        return self.vals['grad_val'][idx]

    def get_grad_vals(self):
        return torch.stack(self.vals['grad_val'])

    def state(self, idx=-1):
        current = {}
        for key, val in self.vals.items():
            try:
                current[key] = val[idx]
            except IndexError:
                print(f'WARNING: chain does not have values for {key}.')
                pass
        return current

    def update(self, state):
        """ Update the chain """
        for key in self.vals.keys():
            self.vals[key].append(state[key])

    def mean(self):
        """ Get the mean of the chain's samples """
        return self.get_samples().mean(0)

    def mc_cov(self, method='inse', adjust=False):
        return st.mc_cov(self.get_samples(), method=method, adjust=adjust)

    def acceptance_rate(self):
        """ proportion of accepted samples """
        return sum(self.vals['accepted'])/len(self.vals['accepted'])

    def multi_ess(self, method='inse', adjust=False):
        return st.multi_ess(self.get_samples(), method=method, adjust=adjust)

    def save(self, path):
        """ Save the chain to disk """
        torch.save(self.vals, path)

    def load(self, path):
        """ Load a previously saved chain """
        self.vals = torch.load(path)

    def to_chainfile(self, keys=None, path=Path.cwd(), mode='a'):
        from .chain_file import ChainFile

        chainfile = ChainFile(keys=keys or self.vals.keys(), path=path, mode=mode)

        for i in range(len(self)):
            chainfile.update(self.state(i), reset=False, close=False)

        chainfile.close()

    def to_kanga(self, keys=None):
        keys = set(keys or self.vals.keys()) & set(['sample', 'target_val', 'grad_val', 'accepted'])

        vals = {}

        for key in keys:
            if key == 'sample':
                vals[key] = self.get_samples().detach().cpu().numpy()
            elif key == 'target_val':
                vals[key] = self.get_target_vals().detach().cpu().numpy()
            elif key == 'grad_val':
                vals[key] = self.get_grad_vals().detach().cpu().numpy()
            elif key == 'accepted':
                vals[key] = np.array(self.vals['accepted'])

        return ChainArray(vals)
