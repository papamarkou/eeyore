# See algorithms 4 and 5 in
# https://arxiv.org/abs/1111.4246

import numpy as np

from .tuner import Tuner

class HMCDATuner(Tuner):
    def __init__(self, l, e0, d=0.65):
        self.l = l
        self.d = d

        sefl.m = np.log(10 * e0)
        self.logbare = 0.
        self.barh = 0.
        self.g = 0.05
        self.t0 = 10
        self.k = 0.75

    def set_d_w(self, iter):
        self.d_w = 1 / (iter + self.t0)

    def set_e_w(self, iter):
        self.e_w = 1 / (iter ** self.k)

    def num_steps(self, e):
        return max(1., round(self.l / e))

    def tune(self, acc_prob, idx, return_e=True):
        iter = idx + 1
        self.set_d_w(iter)
        self.set_e_w(iter)

        self.barh = (1 - self.d_w) * self.barh + self.d_w * (self.d - sampler.acc_prob)
        loge = self.m - np.sqrt(iter) * self.barh / self.g
        self.logbare = self.e_w * loge + (1 - self.e_w) * self.logbare

        if return_e:
            e = np.exp(loge)
            return e, self.num_steps(e)
        else:
            bare = np.exp(self.logbare)
            return bare, self.num_steps(bare)
