import torch

from .xydataset import XYDataset

class EmptyXYDataset(XYDataset):
    def __init__(self, dtype=torch.float64, device='cpu'):
        super().__init__(torch.tensor([[]], dtype=dtype, device=device), torch.tensor([[]], dtype=dtype, device=device))

    def __repr__(self):
        return f'Empty XYDataset'
