import os

import numpy as np

import torch

from eeyore.api import Chain

class ChainFile(Chain):
    """ Monte Carlo chain to store samples in file """

    def __init__(self, keys=['theta', 'target_val', 'accepted'], path=os.getcwd(), mode='a'):
        self.keys = keys
        self.path = path
        self.mode = mode
        self.reset()

    def reset(self):
        self.vals = {key : open(os.path.join(self.path, key+'.csv'), self.mode) for key in self.keys}

    def close(self):
        for key in self.keys:
            self.vals[key].close()

    def update(self, state):
        """ Update the chain """
        self.reset()

        for key in self.keys:
            if isinstance(state[key], torch.Tensor):
                np.savetxt(self.vals[key], state[key].detach().cpu().numpy().ravel()[np.newaxis], delimiter=',')
            elif isinstance(state[key], np.ndarray):
                np.savetxt(self.vals[key], state[key].ravel()[np.newaxis], delimiter=',')
            else:
                self.vals[key].write(str(state[key])+'\n')

        self.close()
