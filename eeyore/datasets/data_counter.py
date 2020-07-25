class DataCounter:
    def __init__(self, batch_size, sample_size,
        num_epochs=None, num_burnin_epochs=None, num_batches=None, drop_last=False):
        self.set_data_info(batch_size, sample_size, num_batches=num_batches, drop_last=drop_last)
        self.set_epoch_info(num_epochs, num_burnin_epochs)
        self.reset()

    def set_num_batches(self, drop_last=False):
        self.num_batches = self.sample_size / self.batch_size
        if (self.sample_size % self.batch_size != 0) and not drop_last:
            self.num_batches = self.num_batches + 1

    def set_data_info(self, batch_size, sample_size, num_batches=None, drop_last=False):
        self.batch_size = batch_size
        self.sample_size = sample_size
        if num_batches is None:
            self.set_num_batches(drop_last=drop_last)
        else:
            self.num_batches = num_batches

    def set_data_info_from_dataloader(self, dataloader):
        self.set_data_info(dataloader.batch_size, len(dataloader.dataset), num_batches=len(dataloader))

    def set_num_iters(self, num_epochs):
        self.num_epochs = num_epochs
        if self.num_epochs is not None:
            self.num_iters = self.num_epochs * self.num_batches
        else:
            self.num_iters = None

    def set_num_burnin_iters(self, num_burnin_epochs):
        self.num_burnin_epochs = num_burnin_epochs
        if self.num_burnin_epochs is not None:
            self.num_burnin_iters = self.num_burnin_epochs * self.num_batches
        else:
            self.num_burnin_iters = None

    def set_epoch_info(self, num_epochs, num_burnin_epochs):
        self.set_num_iters(num_epochs)
        self.set_num_burnin_iters(num_burnin_epochs)

    def set_num_epochs(self, num_iters):
        self.num_iters = num_iters
        if self.num_iters is not None:
            if (self.num_iters % self.num_batches) == 0:
                self.num_epochs = self.num_iters // self.num_batches
            else:
                self.num_epochs = self.num_iters // self.num_batches + 1
        else:
            self.num_epochs = None

    def set_num_burnin_epochs(self, num_burnin_iters):
        self.num_burnin_iters = num_burnin_iters
        if self.num_burnin_iters is not None:
            if (self.num_burnin_iters % self.num_batches) == 0:
                self.num_burnin_epochs = self.num_burnin_iters // self.num_batches
            else:
                self.num_burnin_epochs = self.num_burnin_iters // self.num_batches + 1
        else:
            self.num_burnin_epochs = None

    def set_iter_info(self, num_iters, num_burnin_iters):
        self.set_num_epochs(self, num_iters)
        self.set_num_burnin_epochs(self, num_burnin_iters)

    @classmethod
    def from_dataloader(selfclass, dataloader, num_epochs=None, num_burnin_epochs=None):
        return selfclass(
            dataloader.batch_size,
            len(dataloader.dataset),
            num_epochs=num_epochs,
            num_burnin_epochs=num_burnin_epochs,
            num_batches=len(dataloader)
        )

    def reset(self):
        self.idx = 0

    def increment_idx(self, incr=1):
        self.idx = self.idx + incr
