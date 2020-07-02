from pathlib import Path

from .serial_sampler import SerialSampler

class MultiChainSerialSampler(SerialSampler):
    """ Serial MCMC Sampler with multiple chains"""
    def __init__(self, counter):
        super().__init__(counter=counter)

    def get_sampler(self, i=0):
        return self.samplers[i]
    
    def get_chain(self, i=0):
        return self.get_sampler(i=i).chain

    def set_current(self, theta, data=None):        
        x, y = data or next(iter(self.dataloader))
        for sampler in self.samplers:
            sampler.set_current(theta, data=(x, y))

    def set_all(self, theta, data=None):
        x, y = data or next(iter(self.dataloader))
        for sampler in self.samplers:
            self.set_all(theta, data=(x, y))

    def reset_chains(self):
        for sampler in self.samplers:
            sampler.chain.reset()

    def reset(self, theta, data=None, reset_counter=True, reset_chain=True):
        x, y = data or next(iter(self.dataloader))
        for sampler in self.samplers:
            sampler.reset(theta, data=(x, y), reset_counter=reset_counter, reset_chain=reset_chain)

    def to_chainfile(self, path=Path.cwd(), mode='a'):
        for i, sampler in enumerate(self.samplers):
            sampler.chain.to_chainfile(path=path.joinpath('sampler'+str(i).zfill(self.num_chains)), mode=mode)
