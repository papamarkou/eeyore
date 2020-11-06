# See algorithms 4 and 5 in
# https://arxiv.org/abs/1111.4246

import numpy as np

from .tuner import Tuner

class HMCDATuner(Tuner):
    def __init__(self, l, e0=None, d=0.65, eub=None):
        self.l = l
        self.e0 = e0
        self.d = d
        self.eub = eub # Upper bound for leapfrog step in tuning phase, not in original algorithm

        if self.e0 is None:
            self.m = None
        else:
            self.set_m(self.e0)

        if self.eub is None:
            self.logeub = None
        else:
            self.logeub = np.log(self.eub)

        self.logbare = 0.
        self.barh = 0.
        self.g = 0.05
        self.t0 = 10
        self.k = 0.75

    def set_m(self, e0):
        self.m = np.log(10 * e0)

    def set_d_w(self, iter):
        self.d_w = 1 / (iter + self.t0)

    def set_e_w(self, iter):
        self.e_w = 1 / (iter ** self.k)

    def num_steps(self, e):
        return max(1, round(self.l / e))

    def tune(self, rate, idx, return_e=True):
        iter = idx + 1
        self.set_d_w(iter)
        self.set_e_w(iter)

        self.barh = (1 - self.d_w) * self.barh + self.d_w * (self.d - rate)
        loge = self.m - np.sqrt(iter) * self.barh / self.g
        if self.logeub is not None:
            loge = min(loge, self.logeub)
        self.logbare = self.e_w * loge + (1 - self.e_w) * self.logbare

        if return_e:
            e = np.exp(loge)
            return e, self.num_steps(e)
        else:
            bare = np.exp(self.logbare)
            return bare, self.num_steps(bare)
