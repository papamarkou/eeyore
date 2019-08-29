import torch

from timeit import default_timer as timer
from datetime import timedelta

class Sampler:
    """ Base class for sampling algorithms """

    def draw(self):
        raise NotImplementedError

    def run(self):
        raise NotImplementedError


class SerialSampler(Sampler):
    """ Sequential MCMC Sampler """

    def draw(self):
        raise NotImplementedError

    def run(self, num_iterations, num_burnin, verbose=False):
        """ Run the sampler for num_iterations """
        verbose_msg = "Iteration {:" + str(len(str(num_iterations))) + "}, duration {}"

        for n in range(num_iterations):
            if verbose:
                start_time = timer()

            if n < num_burnin:
                self.draw(savestate=False)
            else:
                self.draw(savestate=True)

            if verbose:
                end_time = timer()
                print(verbose_msg.format(n+1, timedelta(seconds=end_time-start_time)))
