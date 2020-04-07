import numpy as np
import torch

from pathlib import Path

from .chain import Chain
from .chain_list import ChainList
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

    def update(self, state):
        """ Update the chain """
        self.reset()

        for key in self.keys:
            if isinstance(state[key], torch.Tensor):
                np.savetxt(self.vals[key], state[key].detach().cpu().numpy().ravel()[np.newaxis], delimiter=',')
            elif isinstance(state[key], np.ndarray):
                np.savetxt(self.vals[key], state[key].ravel()[np.newaxis], delimiter=',')
            else:
                self.vals[key].write(str(state[key])+'\n')

        self.close()

    def to_chainlist(self, dtype=torch.float64, device='cpu'):
        chainlist_keys = []
        not_converted = []
        chainlist_vals = []

        for key in self.keys:
            if (key == 'sample') or (key == 'grad_val'):
                chainlist_keys.append(key)
                with open(self.path.joinpath(key+'.csv'), mode='r') as file:
                    chainlist_vals.append([
                        torch.tensor(list(map(torch_to_np_types[dtype], line.split(',')))).to(device=device)
                        for line in file.readlines()
                    ])
            elif key == 'target_val':
                chainlist_keys.append(key)
                with open(self.path.joinpath(key+'.csv'), mode='r') as file:
                    chainlist_vals.append([
                        torch.tensor(torch_to_np_types[dtype](line.strip())).to(device=device)
                        for line in file.readlines()
                    ])
            elif key == 'accepted':
                chainlist_keys.append(key)
                with open(self.path.joinpath(key+'.csv'), mode='r') as file:
                    chainlist_vals.append([int(line.strip()) for line in file.readlines()])
            else:
                not_converted.append(key)

        chainlist = ChainList(keys=chainlist_keys, vals=chainlist_vals)

        return chainlist, not_converted
