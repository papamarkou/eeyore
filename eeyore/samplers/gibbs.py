import json
import torch

from .single_chain_serial_sampler import SingleChainSerialSampler
from eeyore.chains import ChainList
from eeyore.datasets import DataCounter
from eeyore.itertools import chunk_evenly
from eeyore.kernels import NormalKernel

class Gibbs(SingleChainSerialSampler):
    def __init__(self, model,
        theta0=None, dataloader=None, data0=None, counter=None,
        scales=1., node_subblock_size=None, chain=ChainList()):
        super(Gibbs, self).__init__(counter or DataCounter.from_dataloader(dataloader))
        self.model = model
        self.dataloader = dataloader

        self.keys = ['sample', 'target_val', 'accepted']
        self.chain = chain

        if theta0 is not None:
            self.set_current(theta0.clone().detach(), data=data0)

        if isinstance(scales, float):
            self.scales = torch.full([self.model.num_par_blocks()], scales, dtype=self.model.dtype, device=self.model.device)
        elif isinstance(scales, torch.Tensor):
            self.scales = scales.to(dtype=self.model.dtype, device=self.model.device)
        elif isinstance(scales, list):
            self.scales = torch.tensor(scales, dtype=self.model.dtype, device=self.model.device)
        else:
            self.scales = scales

        if node_subblock_size is None:
            self.node_subblock_size = [None for _ in range(self.model.num_par_blocks())]
        else:
            self.node_subblock_size = node_subblock_size

    def set_current(self, theta, data=None):
        x, y = super().set_current(theta, data=data)
        self.current['target_val'] = self.model.log_target(self.current['sample'].clone().detach(), x, y)

    def reset(self, theta, data=None, reset_counter=True, reset_chain=True):
        super().reset(theta, data=data, reset_counter=reset_counter, reset_chain=reset_chain)

    def get_blocks(self):
        blocks = []

        for b in range(self.model.num_par_blocks()):
            indices, l, n = self.model.par_block_indices(b)

            if self.node_subblock_size[b] is None:
                indices = [indices]
            else:
                indices = list(chunk_evenly(indices, self.node_subblock_size[b]))

            blocks.append([l, n, indices])

        return blocks

    def save_blocks(self, path='gibbs_lbocks.txt', mode='w'):
        with open(path, mode) as file:
            json.dump(self.get_blocks(), file)

    def draw(self, x, y, savestate=False):
        proposed = {key : None for key in self.keys}
        self.current['accepted'] = []

        if self.counter.num_batches != 1:
            self.current['target_val'] = self.model.log_target(self.current['sample'].clone().detach(), x, y)

        proposed['sample'] = self.current['sample'].clone().detach()

        for b in range(self.model.num_par_blocks()):
            indices, _, _ = self.model.par_block_indices(b)

            if self.node_subblock_size[b] is None:
                indices = [indices]
            else:
                indices = list(chunk_evenly(indices, self.node_subblock_size[b]))

            for i in range(len(indices)):
                kernel = NormalKernel(proposed['sample'][indices[i]], self.scales[b])

                proposed['sample'][indices[i]] = kernel.sample()
                proposed['target_val'] = self.model.log_target(proposed['sample'].clone().detach(), x, y)

                log_rate = proposed['target_val'] - self.current['target_val']
                if torch.log(torch.rand(1, dtype=self.model.dtype, device=self.model.device)) < log_rate:
                    self.current['sample'][indices[i]] = proposed['sample'][indices[i]]
                    self.current['target_val'] = proposed['target_val'].clone().detach()
                    self.current['accepted'].append(1)
                else:
                    self.model.set_params(self.current['sample'].clone().detach())
                    self.current['accepted'].append(0)

        self.current['accepted'] = torch.tensor(self.current['accepted'], device=self.model.device)

        if savestate:
            self.chain.detach_and_update(self.current)

        self.current['sample'].detach_()
        self.current['target_val'].detach_()
