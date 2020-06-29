from pathlib import Path

from .serial_sampler import SerialSampler

class MultiChainSerialSampler(SerialSampler):
    """ Serial MCMC Sampler with multiple chains"""
    def __init__(self, counter):
        super().__init__(counter=counter)

    def set_current(self, theta, data=None):
        self.current = {key : None for key in self.keys}
        self.current['sample'] = theta
        
        x, y = data or next(iter(self.dataloader))
        for sampler in self.samplers:
            sampler.set_current(theta, data=data)

    def reset(self):
        for sampler in self.samplers:
            sampler.reset()

    def to_chainfile(self, path=Path.cwd(), mode='a'):
        for i, sampler in enumerate(self.samplers):
            sampler.chain.to_chainfile(path=path.joinpath('sampler'+str(i).zfill(self.num_chains)), mode=mode)
