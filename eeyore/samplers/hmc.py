import torch

from .single_chain_serial_sampler import SingleChainSerialSampler
from eeyore.chains import ChainList
from eeyore.datasets import DataCounter

class HMC(SingleChainSerialSampler):
    def __init__(self, model,
        theta0=None, dataloader=None, data0=None, counter=None,
        step=0.1, num_steps=10, transform=None, chain=ChainList()):
        super(HMC, self).__init__(counter or DataCounter.from_dataloader(dataloader))
        self.model = model
        self.dataloader = dataloader
        self.step = step
        self.num_steps = num_steps
        self.transform = transform

        self.keys = ['sample', 'target_val', 'grad_val', 'momentum', 'accepted']
        self.chain = chain

        if theta0 is not None:
            self.set_current(theta0.clone().detach(), data=data0)

    def set_current(self, theta, data=None):
        x, y = super().set_current(theta, data=data)
        self.current['target_val'], self.current['grad_val'] = \
            self.model.upto_grad_log_target(self.current['sample'].clone().detach(), x, y)

    def potential_energy(self, position, x, y):
        return -self.model.log_target(position, x, y)

    def upto_grad_potential_energy(self, position, x, y):
        target_val, grad_val = self.model.upto_grad_log_target(position, x, y)
        return -target_val, -grad_val # potential = -target_val, grad_potential = -grad_val

    def log_proposal(self, momentum):
        return - 0.5 * torch.sum(momentum**2)

    def kinetic_energy(self, momentum):
        return -self.log_proposal(momentum)

    def hamiltonian(self, potential, momentum):
        return potential + self.kinetic_energy(momentum)

    def leapfrog(self, position0, momentum0, x, y):
        position = position0.clone().detach()

        # Make a half step for momentum at the beginning
        potential, grad_potential = self.upto_grad_potential_energy(position, x, y)
        momentum = momentum0 - 0.5 * self.step * grad_potential

        # Alternate full steps for position and momentum
        for i in range(self.num_steps-1):
            # Make a full step for the position
            position = position + self.step * momentum

            # Make a full step for the momentum
            potential, grad_potential = self.upto_grad_potential_energy(position.clone().detach(), x, y)
            momentum = momentum - self.step * grad_potential

        # Make a half step for momentum at the end
        position = position + self.step * momentum
        potential, grad_potential = self.upto_grad_potential_energy(position.clone().detach(), x, y)
        momentum = momentum - 0.5 * self.step * grad_potential

        # Negate momentum at end of trajectory to make the proposal symmetric
        momentum = -momentum

        return position, momentum, -potential, -grad_potential # target_val = -potential, grad_val = -grad_potential

    def draw(self, x, y, savestate=False):
        proposed = {key : None for key in self.keys}

        proposed['sample'] = self.current['sample'].clone().detach()

        proposed['momentum'] = torch.randn(self.model.num_params(), dtype=self.model.dtype, device=self.model.device)
        self.current['momentum'] = proposed['momentum'].clone().detach()

        proposed['sample'], proposed['momentum'], proposed['target_val'], proposed['grad_val'] = \
            self.leapfrog(proposed['sample'], proposed['momentum'], x, y)

        log_rate = \
            self.hamiltonian(-self.current['target_val'], self.current['momentum']) \
            - self.hamiltonian(-proposed['target_val'], proposed['momentum'])

        if torch.log(torch.rand(1, dtype=self.model.dtype, device=self.model.device)) < log_rate:
            self.current['sample'] = proposed['sample'].clone().detach()
            self.current['target_val'] = proposed['target_val'].clone().detach()
            self.current['grad_val'] = proposed['grad_val'].clone().detach()
            self.current['accepted'] = 1
        else:
            self.model.set_params(self.current['sample'].clone().detach())
            self.current['accepted'] = 0

        if savestate:
            self.chain.detach_and_update(self.current)

        self.current['sample'].detach_()
        self.current['target_val'].detach_()
        self.current['grad_val'].detach_()
