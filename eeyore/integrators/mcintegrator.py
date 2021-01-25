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
        integral = 0.
        num_kept_samples = 1
        num_dropped_samples = 0

        for sample in self.samples:
            integrand = self.f(sample, x, y)

            if torch.isnan(integrand):
                num_dropped_samples = num_dropped_samples + 1
            else:
                integral = ((num_kept_samples - 1) * integral + integrand) / num_kept_samples
                num_kept_samples = num_kept_samples + 1

        return integral, num_dropped_samples

    def integrate_from_dataset(
        self, dataset, num_points, shuffle=True, dtype=torch.float64, device='cpu', verbose=False, verbose_step=1):
        dataloader = DataLoader(dataset, batch_size=1, shuffle=shuffle)
        counter = DataCounter.from_dataloader(dataloader)

        counter.set_num_epochs(num_points)
        verbose_msg = self.set_verbose_msg(counter)

        integrals = torch.empty(num_points, dtype=dtype, device=device)
        indices = torch.empty(num_points, dtype=torch.int64, device=device)
        nums_dropped_samples = torch.empty(num_points, dtype=torch.int64, device=device)

        for _ in range(counter.num_epochs):
            for _, (x, y, idx) in enumerate(dataloader):
                if counter.idx >= counter.num_iters:
                    break

                if verbose and (((counter.idx+1) % verbose_step) == 0):
                    start_time = timer()

                integral, num_dropped_samples = self.integrate(x, y)
                integrals[counter.idx] = integral.item()
                indices[counter.idx] = idx.clone().detach()
                nums_dropped_samples[counter.idx] = num_dropped_samples

                if verbose and (((counter.idx+1) % verbose_step) == 0):
                    end_time = timer()
                    print(verbose_msg.format(counter.idx+1, timedelta(seconds=end_time-start_time)))

                counter.increment_idx()

        return integrals, indices, nums_dropped_samples

    def set_verbose_msg(self, counter):
        return "Iteration {:" \
            + str(len(str(counter.num_iters))) \
            + "} out of " \
            + str(counter.num_iters) \
            + ", duration {}"
