from datetime import timedelta
from pathlib import Path
from timeit import default_timer as timer

from .sampler import Sampler

class SerialSampler(Sampler):
    """ Serial MCMC Sampler """

    def __init__(self, counter):
        self.counter = counter

    def set_verbose_run_msg(self):
        return 'Iteration {:' \
            + str(len(str(self.counter.num_iters))) \
            + '} out of ' \
            + str(self.counter.num_iters) \
            + ' (in epoch {:' \
            + str(len(str(self.counter.num_epochs))) \
            + '} out of ' \
            + str(self.counter.num_epochs) \
            + '), duration {}'

    def set_verbose_benchmark_msg(self, num_chains):
        return 'Simulating chain {:' \
            + str(len(str(num_chains))) \
            + '} out of ' \
            + str(num_chains) \
            + ' ({:' \
            + str(len(str(num_chains))) \
            + '} failures due to not meeting conditions and {:' \
            + str(len(str(num_chains))) \
            + '} failures due to runtime error)...'

    def run(self, num_epochs, num_burnin_epochs, verbose=False, verbose_step=100):
        """ Run the sampler """

        self.counter.set_epoch_info(num_epochs, num_burnin_epochs)
        verbose_msg = self.set_verbose_run_msg()

        for i in range(self.counter.num_epochs):
            for _, (x, y) in enumerate(self.dataloader):
                if verbose and ((self.counter.idx % verbose_step) == 0):
                    start_time = timer()

                self.draw(x, y, savestate=False if (self.counter.idx < self.counter.num_burnin_iters) else True)

                if verbose and (((self.counter.idx+1) % verbose_step) == 0):
                    end_time = timer()
                    print(verbose_msg.format(self.counter.idx+1, i+1, timedelta(seconds=end_time-start_time)))

                self.counter.increment_idx()

    def benchmark(
        self,
        num_chains,
        num_epochs,
        num_burnin_epochs,
        path,
        check_conditions=None,
        verbose=False,
        verbose_step=100,
        print_acceptance=False,
        print_runtime=True
    ):
        if verbose:
            verbose_msg = self.set_verbose_benchmark_msg(num_chains)

        i, j, k = 0, 0, 0

        while i < num_chains:
            if verbose:
                print(verbose_msg.format(i+1, j, k))

            run_path = Path(path).joinpath('run'+str(i+1).zfill(num_chains))

            try:
                theta0 = self.get_model().prior.sample()
                self.reset(theta0.clone().detach(), data=None, reset_counter=True, reset_chain=True)

                start_time = timer()
                self.run(
                    num_epochs=num_epochs, num_burnin_epochs=num_burnin_epochs, verbose=verbose, verbose_step=verbose_step
                )
                end_time = timer()
                runtime = end_time - start_time

                if ((check_conditions is None) or check_conditions(self.get_chain(), runtime)):
                    self.get_chain().to_chainfile(path=run_path, mode='w')

                    with open(run_path.joinpath('runtime.txt'), 'w') as file:
                        file.write("{}\n".format(runtime))

                    i = i + 1

                    if verbose:
                        print('Succeeded', end='')
                else:
                    j = j + 1

                    if verbose:
                        print('Failed due to not meeting conditions', end='')

                if verbose:
                    if print_acceptance:
                        print('; acceptance rate = {}'.format(self.get_chain().acceptance_rate()), end='')
                    if print_runtime:
                        print('; runtime = {}'.format(timedelta(seconds=runtime)), end='')
                    print('\n')
            except RuntimeError as error:
                with open(run_path.joinpath('errors', 'error'+str(k+1).zfill(num_chains)), 'w') as file:
                    file.write("{}\n".format(error))

                k = k + 1

                if verbose:
                    print('Failed due to runtime error\n')

        with open(Path(path).joinpath('run_counts.txt'), 'w') as file:
            file.write("{},succesful\n".format(i))
            file.write("{},unmet_conditions\n".format(j))
            file.write("{},runtime_errors\n".format(k))
