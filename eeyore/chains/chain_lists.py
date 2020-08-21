class ChainLists:
    def __init__(self, keys=['sample', 'target_val', 'accepted'], vals=None):
        self.reset(keys=keys, vals=vals)

    def reset(self, keys=['sample', 'target_val', 'accepted'], vals=None):
        if vals is None:
            self.vals = {key : [] for key in keys}
        else:
            self.vals = vals

    @classmethod
    def from_chain_list(selfclass, chain_lists, keys=None):
        common_keys = set.intersection(*[set(l) for l in chain_lists])
        class_keys = list(common_keys if keys is None else (set(keys) & common_keys))

        vals = {}
        for key in class_keys:
            vals[key] = [chain_list.vals[key] for chain_list in chain_lists]

        return selfclass(keys=class_keys, vals=vals)
