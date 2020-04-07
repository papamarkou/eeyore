import torch

from datetime import timedelta
from timeit import default_timer as timer
from torch.utils.data import DataLoader

from .integrator import Integrator
from eeyore.datasets import DataCounter

class MCIntegrator(Integrator):
    def __init__(self, f=None, samples=None):
        super(MCIntegrator, self).__init__()
        self.f = f
        self.samples = samples

    def integrate(self, x, y):
        integral = 0
        for sample in self.samples:
            integral = integral + self.f(sample, x, y)
        integral = integral / len(self.samples)

        return integral

    def integrate_from_dataset(
        self, dataset, num_points, shuffle=True, dtype=torch.float64, device='cpu', verbose=False, verbose_step=1):
        dataloader = DataLoader(dataset, batch_size=1, shuffle=shuffle)
        counter = DataCounter.from_dataloader(dataloader)

        counter.set_num_epochs(num_points)
        verbose_msg = self.set_verbose_msg(counter)

        integrals = torch.empty(num_points, dtype=dtype, device=device)
        indices = torch.empty(num_points, dtype=torch.int64, device=device)

        for _ in range(counter.num_epochs):
            for _, (x, y, idx) in enumerate(dataloader):
                if counter.idx >= counter.num_iters:
                    break

                if verbose and (((counter.idx+1) % verbose_step) == 0):
                    start_time = timer()

                integrals[counter.idx] = self.integrate(x, y).item()
                indices[counter.idx] = idx.clone().detach()

                if verbose and (((counter.idx+1) % verbose_step) == 0):
                    end_time = timer()
                    print(verbose_msg.format(counter.idx+1, timedelta(seconds=end_time-start_time)))

                counter.increment_idx()

        return integrals, indices

    def set_verbose_msg(self, counter):
        return "Iteration {:" \
            + str(len(str(counter.num_iters))) \
            + "} out of " \
            + str(counter.num_iters) \
            + ", duration {}"
