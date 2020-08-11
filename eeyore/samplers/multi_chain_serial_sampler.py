from pathlib import Path

from .serial_sampler import SerialSampler

class MultiChainSerialSampler(SerialSampler):
    """ Serial MCMC Sampler with multiple chains"""
    def __init__(self, counter):
        super().__init__(counter=counter)

    def default_indicator(self):
        return 0

    def get_model(self, idx=None):
        return self.samplers[idx or self.default_indicator()].model

    def get_chain(self, idx=None):
        return self.samplers[idx or self.default_indicator()].chain

    def get_sample(self, param_idx, chain_idx=None):
        return self.get_chain(idx=chain_idx).get_sample(idx=param_idx)

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
            sampler.chain.reset(keys=sampler.chain.vals.keys())

    def reset(self, theta, data=None, reset_counter=True, reset_chain=True):
        x, y = data or next(iter(self.dataloader))
        for sampler in self.samplers:
            sampler.reset(theta, data=(x, y), reset_counter=reset_counter, reset_chain=reset_chain)

    def to_chainfile(self, path=Path.cwd(), mode='a'):
        for i, sampler in enumerate(self.samplers):
            sampler.chain.to_chainfile(path=path.joinpath('sampler'+str(i).zfill(self.num_chains)), mode=mode)
