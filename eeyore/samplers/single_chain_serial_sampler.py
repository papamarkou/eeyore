from pathlib import Path

from .serial_sampler import SerialSampler

class SingleChainSerialSampler(SerialSampler):
    """ Serial MCMC Sampler with single chain"""
    def __init__(self, counter):
        super().__init__(counter=counter)

    def set_current(self, theta, data=None):
        self.current = {key : None for key in self.keys}
        self.current['sample'] = theta
        
        x, y = data or next(iter(self.dataloader))

        return x, y

    def set_all(self, theta, data=None):
        self.set_current(theta, data=data)

    def reset(self, theta, data=None, reset_counter=True, reset_chain=True):
        if reset_counter:
            self.counter.reset()
        if reset_chain:
            self.chain.reset()
        self.set_all(theta, data=data)

    def to_chainfile(self, path=Path.cwd(), mode='a'):
        self.chain.to_chainfile(path=path, mode=mode)
