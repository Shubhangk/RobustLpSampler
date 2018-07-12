import math
import random
import numpy as np


def p_stableDistribution(p):
    r = np.random.random()

    PI = math.pi
    theta = random.uniform(-PI / 2.0, PI / 2.0)

    # convenience notation
    Xl = math.sin(p * theta) / (math.cos(theta) ** (1 / float(p)))

    Xr = math.cos(theta * (1 - p)) / math.log(r, math.e)
    Xr = Xr ** ((1-p)/p)

    X = Xl * Xr
    return X

class LpEstimator:
    A = []
    r = -1
    y = []

    def initialize(self, n, p, eps):
        CONSTANT = 100
        self.r = CONSTANT * (1 / float(eps))

        for i in range(int(self.r)):
            self.A.append([p_stableDistribution(p) for _ in range(n)])

        self.y = np.multiply(self.A, [0 for _ in range(n)])

    def update(self, i):
        for j in range(len(self.y)):
            self.y[j] += self.A[j][i]


    def output(self):
        assert len(self.y) == self.r
        temp = [abs(self.y[i]) for i in range(len(self.y))]
        median_y = np.median(temp)

        MEDIAN_Dp = -1.891000000000012
        return float(median_y) / abs(float(MEDIAN_Dp))

## driver to estimate median of Dp according to the p-stable distribution above
x = []
delta = 0.001
j = -2
while j < 1:
    x.append(j)
    j += delta
#print(x)

for i in range(len(x)):
    try:
        #if i % 10 == 0:
            #print(i)
        temp = p_stableDistribution(x[i])
        #print(temp)
        if 5.0/8.0 < temp < 3.0/4.0:
            print("x : " + str(x[i]))
            print("f(x) : " + str(temp))
            print("------")
    except:
        #print(i)
        continue