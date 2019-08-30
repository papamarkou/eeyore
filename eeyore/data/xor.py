import torch
from torch.utils.data import Dataset


class XOR(Dataset):

    def __init__(self, dtype=torch.float64, device='cpu'):
        self.dtype = dtype
        self.device = device
        self.load_data()

    def __repr__(self):
        return f'XOR dataset'

    def __len__(self):
        return len(self.data)

    def load_data(self):
        self.data = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=self.dtype, device=self.device)

        self.labels = torch.tensor([[0], [1], [1], [0]], dtype=self.dtype, device=self.device)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
