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
