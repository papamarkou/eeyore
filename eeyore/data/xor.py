import torch
from torch.utils.data import Dataset


class XOR(Dataset):

    def __init__(self, dtype=torch.float64):
        self.dtype = dtype
        self.load_data()

    def __repr__(self):
        return f'XOR dataset'

    def __len__(self):
        return len(self.data)

    def load_data(self):
        self.data = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=self.dtype)
        self.labels = torch.tensor([[0], [1], [1], [0]], dtype=self.dtype)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
