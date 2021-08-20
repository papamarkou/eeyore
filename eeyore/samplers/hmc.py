import torch

from .single_chain_serial_sampler import SingleChainSerialSampler
from eeyore.chains import ChainList
from eeyore.datasets import DataCounter
from eeyore.tuners import HMCDATuner

class HMC(SingleChainSerialSampler):
    def __init__(self, model,
        theta0=None, dataloader=None, data0=None, counter=None,
        step=0.1, num_steps=10, tuner=None, chain=ChainList()):
        super(HMC, self).__init__(counter or DataCounter.from_dataloader(dataloader))
        self.model = model
        self.dataloader = dataloader
        self.tuner = tuner

        if self.tuner is not None:
            if isinstance(self.tuner, HMCDATuner):
                if self.tuner.e0 is None:
                    self.init_step(theta0.clone().detach())
                    if self.tuner.eub is not None:
                        self.step = min(self.tuner.eub, self.step)
                    self.tuner.set_m(self.step)
                else:
                    self.step = self.tuner.e0

                self.num_steps = self.tuner.num_steps(self.step)
        else:
            self.step = step
            self.num_steps = num_steps

        self.keys = ['sample', 'target_val', 'grad_val', 'momentum', 'hamiltonian', 'accepted']
        self.chain = chain

        if theta0 is not None:
            self.set_current(theta0.clone().detach(), data=data0)

    def init_step(self, theta):
        iterator = iter(self.dataloader)
        x, y = next(iterator)

        self.step = 1.
        self.num_steps = 1

        current = {
            'sample': theta,
            'momentum': torch.randn(self.model.num_params(), dtype=self.model.dtype, device=self.model.device)
        }
        current['target_val'] = self.model.log_target(current['sample'].clone().detach(), x, y)
        current['hamiltonian'] = self.hamiltonian(-current['target_val'], current['momentum'])

        proposed = {}
        proposed['sample'], proposed['momentum'], proposed['target_val'], _ = \
            self.leapfrog(current['sample'], current['momentum'], x, y)
        proposed['hamiltonian'] = self.hamiltonian(-proposed['target_val'], proposed['momentum'])

        ratio = torch.exp(current['hamiltonian'] - proposed['hamiltonian'])

        a = 2 * (ratio > 0.5) - 1

        while torch.pow(ratio, a) > torch.pow(2, -a):
            try:
                x, y = next(iterator)
            except StopIteration:
                iterator = iter(self.dataloader)
                x, y = next(iterator)

            self.step = (torch.pow(2, a) * self.step).item()

            current['target_val'] = self.model.log_target(current['sample'].clone().detach(), x, y)
            current['hamiltonian'] = self.hamiltonian(-current['target_val'], current['momentum'])

            _, proposed['momentum'], proposed['target_val'], _ = \
                self.leapfrog(current['sample'], current['momentum'], x, y)
            proposed['hamiltonian'] = self.hamiltonian(-proposed['target_val'], proposed['momentum'])

            ratio = torch.exp(current['hamiltonian'] - proposed['hamiltonian'])

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

        if self.counter.num_batches != 1:
            self.current['target_val'], self.current['grad_val'] = \
                self.model.upto_grad_log_target(self.current['sample'].clone().detach(), x, y)

        proposed['sample'] = self.current['sample'].clone().detach()
        proposed['momentum'] = torch.randn(self.model.num_params(), dtype=self.model.dtype, device=self.model.device)

        self.current['momentum'] = proposed['momentum'].clone().detach()
        self.current['hamiltonian'] = self.hamiltonian(-self.current['target_val'], self.current['momentum'])

        proposed['sample'], proposed['momentum'], proposed['target_val'], proposed['grad_val'] = \
            self.leapfrog(proposed['sample'], proposed['momentum'], x, y)
        proposed['hamiltonian'] = self.hamiltonian(-proposed['target_val'], proposed['momentum'])

        rate = torch.min(
            torch.exp(self.current['hamiltonian'] - proposed['hamiltonian']),
            torch.tensor([1.], dtype=self.model.dtype, device=self.model.device)
        )

        if torch.rand(1, dtype=self.model.dtype, device=self.model.device) < rate:
            self.current['sample'] = proposed['sample'].clone().detach()
            if self.counter.num_batches == 1:
                self.current['target_val'] = proposed['target_val'].clone().detach()
                self.current['grad_val'] = proposed['grad_val'].clone().detach()
            self.current['accepted'] = 1
        else:
            self.model.set_params(self.current['sample'].clone().detach())
            self.current['accepted'] = 0

        if self.tuner is not None:
            if isinstance(self.tuner, HMCDATuner):
                if self.counter.idx < self.counter.num_burnin_iters:
                    self.step, self.num_steps = self.tuner.tune(
                        rate.item(), self.counter.idx, return_e=self.counter.idx != self.counter.num_burnin_iters - 1
                    )

        if savestate:
            self.chain.detach_and_update(self.current)

        self.current['sample'].detach_()
        self.current['target_val'].detach_()
        self.current['grad_val'].detach_()
