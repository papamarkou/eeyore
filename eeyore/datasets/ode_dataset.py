from .xydataset import XYDataset

class ODEDataset(XYDataset):
    def __init__(self, t, y):
        super().__init__(t, y)
        self.reshape()

    def reshape(self):
        if len(self.x.shape) == 1:
            self.x = self.x.view(len(self.x), 1)
        if len(self.y.shape) == 1:
            self.y = self.y.view(len(self.y), 1)
