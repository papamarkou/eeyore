import numpy as np
import torch

from pathlib import Path

from .chain import Chain
from eeyore.constants import torch_to_np_types

class ChainFile(Chain):
    """ Monte Carlo chain to store samples in file """

    def __init__(self, keys=['sample', 'target_val', 'accepted'], path=Path.cwd(), mode='a'):
        self.keys = keys
        self.path = path
        self.mode = mode

        if not self.path.exists():
            self.path.mkdir(parents=True, exist_ok=True)

        self.reset()

    def reset(self):
        self.vals = {key : open(self.path.joinpath(key+'.csv'), self.mode) for key in self.keys}

    def close(self):
        for key in self.keys:
            self.vals[key].close()

    def update(self, state, reset=True, close=True):
        """ Update the chain """
        if reset:
            self.reset()

        for key in self.keys:
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
        if key == 'accepted':
            return self.scalar_line_to_val_element(line, dtype=int)
        elif key == 'target_val':
            return self.singleton_line_to_val_element(line, dtype=dtype, device=device)
        elif (key == 'sample') or (key == 'grad_val'):
            return self.vector_line_to_val_element(line, dtype=dtype, device=device)

    def to_chainlist(self, dtype=torch.float64, device='cpu'):
        from .chain_list import ChainList

        chainlist_keys = []
        not_converted = []
        chainlist_vals = []

        for key in self.keys:
            if key in ('accepted', 'target_val', 'sample', 'grad_val'):
                chainlist_keys.append(key)
                with open(self.path.joinpath(key+'.csv'), mode='r') as file:
                    chainlist_vals.append([
                        self.line_to_val_element(line, key, dtype=dtype, device=device) for line in file.readlines()
                    ])
            else:
                not_converted.append(key)

        chainlist = ChainList(keys=chainlist_keys, vals=chainlist_vals)

        return chainlist, not_converted
