# https://arxiv.org/abs/1111.4246

import numpy as np

class HMCDATuner(Tuner):
    def __init__(self, l, e0, d=0.65):
        self.l = l
        self.e = e0
        self.d = d

        sefl.m = np.log(10 * e0)
        self.logbare = 0.
        self.barh = 0.
        self.g = 0.05
        self.t0 = 10
        self.k = 0.75

        self.loge = np.log(self.e)

    def set_iter(self, idx):
        self.iter = idx + 1

    def set_d_w(self):
        self.d_w = 1 / (self.iter + self.t0)

    def set_e_w(self):
        self.e_w = 1 / (self.iter ** self.k)

    def tune(sampler):
        if sampler.counter.idx < sampler.counter.num_burnin_iters:
            self.set_iter(sampler.counter.idx)
            self.set_d_w()
            self.set_e_w()

            self.barh = (1 - self.d_w) * self.barh + self.d_w * (self.d - sampler.acc_prob)
            self.loge = self.m - np.sqrt(self.iter) * self.barh / self.g
            self.e = np.exp(self.loge)
            self.logbare = self.e_w * self.loge + (1 - self.e_w) * self.logbare

            self.n = max(1., round(self.l / self.e))

            return self.e, self.n
        else:
            return sampler.step, sampler.num_steps
