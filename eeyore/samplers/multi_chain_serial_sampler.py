from pathlib import Path

from .serial_sampler import SerialSampler

class MultiChainSerialSampler(SerialSampler):
    """ Serial MCMC Sampler with multiple chains"""
    def __init__(self, counter):
        super().__init__(counter=counter)
    
    def reset(self):
        for sampler in self.samplers:
            sampler.reset()

    def to_chainfile(self, path=Path.cwd(), mode='a'):
        for i, sampler in enumerate(self.samplers):
            sampler.chain.to_chainfile(path=path.joinpath('sampler'+str(i).zfill(self.num_chains)), mode=mode)
