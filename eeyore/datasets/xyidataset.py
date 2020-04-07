from .xydataset import XYDataset

class XYIDataset(XYDataset):
    def __init__(self, x, y):
        super().__init__(x, y)

    def __repr__(self):
        return f'XYIDataset: indexed XYDataset'

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx], idx

    @classmethod
    def from_xydataset(selfclass, xydataset):
        return selfclass(xydataset.x, xydataset.y)
