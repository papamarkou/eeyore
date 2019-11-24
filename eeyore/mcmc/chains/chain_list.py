import torch

from eeyore.api import Chain

class ChainList(Chain):
    """ Monte Carlo chain to store samples in lists """

    def __init__(self, keys=['theta', 'target_val', 'accepted'], vals=None):
        self.keys = keys
        self.reset(vals=vals)

    def __repr__(self):
        return f"Markov chain containing {len(self.vals['theta'])} samples."

    def __len__(self):
        return len(self.vals['theta'])

    def reset(self, vals=None):
        if vals is None:
            self.vals = {key : [] for key in self.keys}
        else:
            self.vals = dict(zip(self.keys, vals))

    def get_theta(self, i):
        return [theta[i].item() for theta in self.vals['theta']]

    def get_target_vals(self):
        return [target_val.item() for target_val in self.vals['target_val']]

    def state(self):
        current = {}
        for key, val in self.vals.items():
            try:
                current[key] = val[-1]
            except IndexError:
                print(f'WARNING: chain does not have values for {key}.')
                pass
        return current

    def update(self, state):
        """ Update the chain """
        for key in self.keys:
            self.vals[key].append(state[key])

    def mean(self):
        """ Get the mean of the chain's samples """
        samples = torch.stack(self.vals['theta'])
        return samples.mean(0)

    def acceptance_rate(self):
        """ proportion of accepted samples """
        accepted = torch.tensor(self.vals['accepted'], dtype=torch.float)
        return accepted.mean().item()

    def save(self, path):
        """ Save the chain to disk """
        torch.save(self.vals, path)

    def load(self, path):
        """ Load a previously saved chain """
        self.vals = torch.load(path)
