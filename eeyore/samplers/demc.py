import copy
import numpy as np

from pathlib import Path

from .metropolis_hastings import MetropolisHastings
from .multi_chain_serial_sampler import MultiChainSerialSampler
from eeyore.chains import ChainFile, ChainList
from eeyore.datasets import DataCounter
from eeyore.kernels import DEMCKernel
from eeyore.stats import choose_from_subset

class DEMC(MultiChainSerialSampler):
    def __init__(self, model, sigmas, dataloader,
        theta0=None, data0=None, counter=None,
        num_chains=10, c=None, schedule=lambda n, num_iterations: 1., storage='list',
        keys=['sample', 'target_val', 'accepted'], path=Path.cwd(), mode='a'):
        super(DEMC, self).__init__(counter or DataCounter.from_dataloader(dataloader))
        self.sigmas = sigmas
        self.dataloader = dataloader
        self.num_chains = num_chains
        self.schedule = schedule

        self.c = c or [2.38/np.sqrt(model.num_params()) for i in range(self.num_chains)]

        self.init_samplers(model, theta0, data0 or next(iter(self.dataloader)), storage, keys, path, mode)

    def init_kernel(self, i, model):
        kernel = DEMCKernel(c=self.c[i])
        kernel.init_a_and_b(model.num_params(), model.dtype, model.device)
        kernel.init_density(model.num_params(), model.dtype, model.device)
        kernel.density.scale = self.sigmas[i]

        return kernel

    def init_chain(self, i, storage, keys, path, mode):
        if storage == 'list':
            chain = ChainList(keys=keys)
        elif storage == 'file':
            chain_path = path.joinpath('chain'+f"{(i+1):0{len(str(self.num_chains))}}")
            if not chain_path.exists():
                chain_path.mkdir(parents=True, exist_ok=True)
            chain = ChainFile(keys=keys, path=chain_path, mode=mode)

        return chain

    def init_samplers(self, model, theta0, data0, storage, keys, path, mode):
        self.samplers = []
        for i in range(self.num_chains):
            self.samplers.append(MetropolisHastings(
                copy.deepcopy(model),
                theta0=theta0, dataloader=None, data0=data0, counter=self.counter,
                symmetric=True, kernel=self.init_kernel(i, model), chain=self.init_chain(i, storage, keys, path, mode)
            ))

    def set_temperature(self, n, num_iterations):
        temperature = self.schedule(n, num_iterations)
        for i in range(self.num_chains):
            self.samplers[i].model.temperature = temperature

    def set_kernel(self, i):
        j = choose_from_subset(self.num_chains, [i])
        k = choose_from_subset(self.num_chains, [i, j])
        self.samplers[i].kernel.set_a_and_b(
            self.samplers[j].current['sample'].clone().detach(), self.samplers[k].current['sample'].clone().detach()
        )
        self.samplers[i].kernel.set_density_params(self.samplers[i].current['sample'].clone().detach())

    def reset(self, theta, data=None):
        super().reset(theta, data=data, reset_counter=False, reset_chain=True)
        self.counter.reset()

    def draw(self, x, y, savestate=False):
        self.set_temperature(self.counter.idx, self.counter.num_iters)

        for i in range(self.num_chains):
            self.set_kernel(i)
            self.samplers[i].draw(x, y, savestate=savestate)
