import random
import numpy as np

class AMS_Estimator:
    m = 0
    r = 0
    a = 0

    def Initialize(self):
        self.m = 0
        self.r = 0
        self.a = 0

    def Process(self, j):
        self.m += 1
        p = 1 / float(self.m)
        B = np.random.choice([0,1], [1-p, p])

        if B == 1:
            self.a = j
            self.r = 0

        if j == self.a:
            self.r += 1

    def Output(self, k):
        m = self.m
        r = self.r
        a = self.a
        return m * (r ** k - (r - 1) ** k)
