from pathlib import Path

from .serial_sampler import SerialSampler

class SingleChainSerialSampler(SerialSampler):
    """ Serial MCMC Sampler with single chain"""
    def __init__(self, counter):
        super().__init__(counter=counter)

    def set_current(self, theta, data=None):
        x, y = data or next(iter(self.dataloader))
        self.current['sample'] = theta
        
        return x, y

    def reset(self):
        self.counter.reset()
        self.chain.reset()

    def to_chainfile(self, path=Path.cwd(), mode='a'):
        self.chain.to_chainfile(path=path, mode=mode)
