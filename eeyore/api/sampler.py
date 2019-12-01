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

    def draw(self, n):
        raise NotImplementedError

    def run(self, num_iterations, num_burnin, verbose=False, verbose_step=100):
        """ Run the sampler for num_iterations """
        verbose_msg = "Iteration {:" + str(len(str(num_iterations))) + "}, duration {}"

        for n in range(num_iterations):
            if verbose and (((n+1) % verbose_step) == 0):
                start_time = timer()

            if n < num_burnin:
                self.draw(n, savestate=False)
            else:
                self.draw(n, savestate=True)

            if verbose and (((n+1) % verbose_step) == 0):
                end_time = timer()
                print(verbose_msg.format(n+1, timedelta(seconds=end_time-start_time)))
