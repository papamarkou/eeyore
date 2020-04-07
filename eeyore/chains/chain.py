import torch

class Chain:
    """ Base class for Monte Carlo chains """

    def reset(self):
        raise NotImplementedError

    def update(self, state):
        raise NotImplementedError

    def detach_and_update(self, state):
        self.update({k: v.clone().detach() if isinstance(v, torch.Tensor) else v for k, v in state.items()})
