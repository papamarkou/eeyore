import numpy as np
import torch

from pathlib import Path
from torch.nn.functional import one_hot
from torch.utils.data import Dataset

from .data_info import data_paths
from eeyore.constants import torch_to_np_types

class XYDataset(Dataset):
    def __init__(self, x, y):
        self.set_data(x, y)

    def __repr__(self):
        return f'XYDataset'

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    def set_data(self, x, y):
        self.x = x
        self.y = y

    @classmethod
    def from_file(selfclass, path=Path.cwd(), xfile='x.csv', yfile='y.csv', xskiprows=1, yskiprows=1, xusecols=None,
        yusecols=None, xndmin=2, yndmin=2, dtype=torch.float64, device='cpu', xonehot=False, yonehot=False):
        x = torch.from_numpy(np.loadtxt(
            path.joinpath(xfile),
            dtype=torch_to_np_types[dtype], delimiter=',', skiprows=xskiprows, usecols=xusecols, ndmin=xndmin
        )).to(device=device)
        if xonehot:
            x = one_hot(x.long()).to(x.dtype)

        y = torch.from_numpy(np.loadtxt(
            path.joinpath(yfile),
            dtype=torch_to_np_types[dtype], delimiter=',', skiprows=yskiprows, usecols=yusecols, ndmin=yndmin
        )).to(device=device)
        if yonehot:
            y = one_hot(y.long()).to(y.dtype)

        return selfclass(x, y)

    @classmethod
    def from_eeyore(selfclass, data_name,
        xndmin=2, yndmin=2, dtype=torch.float64, device='cpu', xonehot=False, yonehot=False):
        return selfclass.from_file(
            path=data_paths[data_name],
            xndmin=xndmin, yndmin=yndmin, dtype=dtype, device=device, xonehot=xonehot, yonehot=yonehot
        )
