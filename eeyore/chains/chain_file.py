import numpy as np
import torch

from pathlib import Path

from .chain import Chain
from eeyore.constants import torch_to_np_types

class ChainFile(Chain):
    """ Monte Carlo chain to store samples in file """

    def __init__(self, keys=['sample', 'target_val', 'accepted'], path=Path.cwd(), mode='a'):
        self.path = path
        self.mode = mode

        if not self.path.exists():
            self.path.mkdir(parents=True, exist_ok=True)

        self.reset(keys=keys)

    def reset(self, keys=['sample', 'target_val', 'accepted']):
        self.vals = {key : open(self.path.joinpath(key+'.csv'), self.mode) for key in keys}

    def close(self):
        for key in self.vals.keys():
            self.vals[key].close()

    def update(self, state, reset=True, close=True):
        """ Update the chain """
        if reset:
            self.reset(keys=self.vals.keys())

        for key in self.vals.keys():
            if isinstance(state[key], torch.Tensor):
                np.savetxt(self.vals[key], state[key].detach().cpu().numpy().ravel()[np.newaxis], delimiter=',')
            elif isinstance(state[key], np.ndarray):
                np.savetxt(self.vals[key], state[key].ravel()[np.newaxis], delimiter=',')
            else:
                self.vals[key].write(str(state[key])+'\n')

        if close:
            self.close()

    def scalar_line_to_val_element(self, line, dtype=int):
        return dtype(line.strip())

    def singleton_line_to_val_element(self, line, dtype=torch.float64, device='cpu'):
        return torch.tensor(torch_to_np_types[dtype](line.strip())).to(device=device)

    def vector_line_to_val_element(self, line, dtype=torch.float64, device='cpu'):
        return torch.tensor(list(map(torch_to_np_types[dtype], line.split(',')))).to(device=device)

    def line_to_val_element(self, line, key, dtype=torch.float64, device='cpu'):
        if key == 'target_val':
            return self.singleton_line_to_val_element(line, dtype=dtype, device=device)
        elif (key == 'sample') or (key == 'grad_val'):
            return self.vector_line_to_val_element(line, dtype=dtype, device=device)
        elif key == 'accepted':
            return self.scalar_line_to_val_element(line, dtype=int)

    def to_chainlist(self, keys=None, dtype=torch.float64, device='cpu'):
        from .chain_list import ChainList

        keys = set(keys or self.vals.keys()) & set(['sample', 'target_val', 'grad_val', 'accepted'])

        chainlist_keys = []
        chainlist_vals = []

        for key in keys:
            chainlist_keys.append(key)
            with open(self.path.joinpath(key+'.csv'), mode='r') as file:
                chainlist_vals.append([
                    self.line_to_val_element(line, key, dtype=dtype, device=device) for line in file.readlines()
                ])

        chainlist = ChainList(vals=dict(zip(chainlist_keys, chainlist_vals)))

        return chainlist
