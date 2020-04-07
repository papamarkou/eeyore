from datetime import timedelta
from timeit import default_timer as timer

from .sampler import Sampler

class SerialSampler(Sampler):
    """ Sequential MCMC Sampler """

    def __init__(self, counter):
        self.counter = counter

    def run(self, num_epochs, num_burnin_epochs, verbose=False, verbose_step=100):
        """ Run the sampler """

        self.counter.set_epoch_info(num_epochs, num_burnin_epochs)
        verbose_msg = self.set_verbose_msg()

        for i in range(self.counter.num_epochs):
            for _, (x, y) in enumerate(self.dataloader):
                if verbose and (((self.counter.idx+1) % verbose_step) == 0):
                    start_time = timer()

                self.draw(x, y, savestate=False if (self.counter.idx < self.counter.num_burnin_iters) else True)

                if verbose and (((self.counter.idx+1) % verbose_step) == 0):
                    end_time = timer()
                    print(verbose_msg.format(self.counter.idx+1, i+1, timedelta(seconds=end_time-start_time)))

                self.counter.increment_idx()

    def set_verbose_msg(self):
        return "Iteration {:" \
            + str(len(str(self.counter.num_iters))) \
            + "} out of " \
            + str(self.counter.num_iters) \
            + " (in epoch {:" \
            + str(len(str(self.counter.num_epochs))) \
            + "} out of " \
            + str(self.counter.num_epochs) \
            + "), duration {}"
