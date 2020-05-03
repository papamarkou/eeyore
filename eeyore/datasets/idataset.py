from torch.utils.data import Dataset

class IDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __repr__(self):
        return f'IDataset: indexed Dataset'

    def __len__(self):
        return self.dataset.__len__()

    def __getitem__(self, idx):
        x, y = self.dataset.__getitem__(idx)
        return x, y, idx
