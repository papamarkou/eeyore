import torch

from .chain_file import ChainFile

class ChainLists:
    def __init__(self, keys=['sample', 'target_val', 'accepted'], vals=None):
        self.reset(keys=keys, vals=vals)

    def reset(self, keys=['sample', 'target_val', 'accepted'], vals=None):
        if vals is None:
            self.vals = {key : [] for key in keys}
        else:
            self.vals = vals

    @classmethod
    def from_chain_list(selfclass, chain_lists, keys=['sample', 'target_val', 'accepted']):
        common_keys = set.intersection(*[set(chain_list.vals.keys()) for chain_list in chain_lists])
        class_keys = set(keys) & common_keys

        vals = {}

        for key in class_keys:
            vals[key] = [chain_list.vals[key] for chain_list in chain_lists]

        return selfclass(keys=class_keys, vals=vals)

    @classmethod
    def from_file(selfclass, paths, keys=['sample', 'target_val', 'accepted'], mode='a', dtype=torch.float64, device='cpu'):
        chain_lists = []

        for path in paths:
            chain_lists.append(ChainFile(keys=keys, path=path, mode=mode).to_chainlist(dtype=dtype, device=device))

        return selfclass.from_chain_list(chain_lists, keys=keys)

    def __repr__(self):
        return f"{len(self)} Markov chains, each containing {self.num_samples()} samples."

    def __len__(self):
        return self.num_chains()

    def num_params(self):
        return len(self.vals['sample'][0][0])

    def num_samples(self):
        return len(self.vals['sample'][0])

    def num_chains(self):
        return len(self.vals['sample'])

    def get_samples(self):
        return torch.stack([self.get_chain(i) for i in range(self.num_chains())])

    def get_target_vals(self):
        return torch.stack([self.get_chain(i, key='target_val') for i in range(self.num_chains())])

    def get_grad_vals(self):
        return torch.stack([self.get_chain(i, key='grad_val') for i in range(self.num_chains())])

    def get_chain(self, idx, key='sample'):
        return torch.stack(self.vals[key][idx])

    def mean(self):
        return self.get_samples().mean(1)
