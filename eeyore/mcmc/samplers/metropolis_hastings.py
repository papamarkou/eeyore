import torch

from eeyore.api import SerialSampler
from eeyore.mcmc import MCChain


class MetropolisHastings(SerialSampler):

    def __init__(self, model, kernel, theta0, dataloader, keys=['theta', 'target_val', 'accepted']):
        super(MetropolisHastings, self).__init__()
        self.model = model
        self.kernel = kernel
        self.dataloader = dataloader

        self.keys = ['theta', 'target_val']
        self.current = {key : None for key in self.keys}
        self.chain = MCChain(keys)

        self.reset(theta0)

    def reset(self, theta):
        data, label = next(iter(self.dataloader))

        self.current['theta'] = theta.clone().detach()
        self.current['target_val'] = self.model.log_target(self.current['theta'], data, label)

    def draw(self, savestate=False):
        proposed = {key : None for key in self.keys}

        for data, label in self.dataloader:
            log_rate = 0
            proposed['theta'] = self.kernel.sample()
            proposed['target_val'] = self.model.log_target(proposed['theta'], data, label)
            log_rate = log_rate - self.current['target_val'] - self.kernel.log_density(proposed['theta'])
            self.kernel.set_density(proposed['theta'])
            log_rate = log_rate + proposed['target_val'] + self.kernel.log_density(self.current['theta'])

            threshold = torch.rand(1)
            if torch.log(threshold) < log_rate:
                self.current['theta'] = proposed['theta']
                self.current['target_val'] = proposed['target_val']
                self.current['accepted'] = 1
            else:
                self.model.set_params(self.current['theta'])
                self.kernel.set_density(self.current['theta'])
                self.current['accepted'] = 0

            if savestate:
                self.chain.update(self.current)
