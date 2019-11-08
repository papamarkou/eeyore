class Chain:
    """ Base class for Monte Carlo chains """

    def reset(self):
        raise NotImplementedError

    def update(self, state):
        raise NotImplementedError
