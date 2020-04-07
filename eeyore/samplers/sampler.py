class Sampler:
    """ Base class for sampling algorithms """

    def draw(self, x, y, savestate=False):
        raise NotImplementedError

    def run(self, num_epochs, num_burnin_epochs, verbose=False, verbose_step=100):
        raise NotImplementedError
