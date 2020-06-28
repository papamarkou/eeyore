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
                if verbose and (((self.counter.idx+1) % verbose_step) == 0):
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
        verbose=True,
        verbose_step=100
        # chain_basename='chain',
        # accepted_basename='accepted',
        # runtime_basename='runtime',
        # print_acceptance=True,
        # print_runtime=True,
        # error_basename='error'
    ):
        if not Path(path).exists():
            Path(path).mkdir(parents=True, exist_ok=True)

        if verbose:
            verbose_msg = self.set_verbose_benchmark_msg(num_chains)

        i, j, k = 0, 0, 0

        while i < num_chains:
            if verbose:
                print(verbose_msg.format(i+1, j, k))

            try:
                theta0 = model.prior.sample()
                sampler.reset()
                # sampler = generate_sampler(model, theta0, dataloader)

                start_time = timer()
                sampler.run(
                    num_epochs=num_epochs, num_burnin_epochs=num_burnin_epochs, verbose=verbose, verbose_step=verbose_step
                )
                end_time = timer()
                runtime = end_time - start_time

                if ((check_conditions is None) or check_conditions(get_chain(sampler), runtime)):
                    chain = torch.empty(num_post_burnin_epochs, model.num_params())
                    for l in range(model.num_params()):
                        chain[:, l] = torch.tensor(get_chain(sampler).get_sample(k))

                    with open(Path(outpath).joinpath('{}{:02d}.csv'.format(chain_basename, i+1)), 'w') as file:
                        np.savetxt(file, chain.cpu().detach().numpy(), delimiter=',', newline='\n', header='')

                    with open(Path(outpath).joinpath('{}{:02d}.txt'.format(accepted_basename, i+1)), 'w') as file:
                        writer = csv.writer(file)
                        for a in get_chain(sampler).vals['accepted']:
                            writer.writerow([a])

                    with open(Path(outpath).joinpath('{}{:02d}.txt'.format(runtime_basename, i+1)), 'w') as file:
                        file.write(str('{}'.format(runtime)))
                        file.write('\n')

                    i = i + 1

                    if verbose:
                        print('Succeeded', end='')
                else:
                    j = j + 1

                    if verbose:
                        print('Failed due to not meeting conditions', end='')

                if verbose:
                    if print_acceptance:
                        print('; acceptance rate = {}'.format(get_chain(sampler).acceptance_rate()), end='')
                    if print_runtime:
                        print('; runtime = {}'.format(timedelta(seconds=runtime)), end='')
                    print('\n')
            except RuntimeError as error:
                with open(Path(outpath).joinpath('{}{:02d}.txt'.format(error_basename, k+1)), 'w') as file:
                    file.write(error)
                    file.write('\n')

                k = k + 1

                if verbose:
                    print('Failed due to runtime error\n')
