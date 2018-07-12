import random
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.stats import levy_stable

class Tug_of_War:
    h = {}
    x = 0

    def Initialize(self):
        self.h = {}
        self.x = 0

    def Process(self, j, c):
        if self.h.get(str(j)) is None:
            self.h[str(j)] = np.random.choice([-1, 1])

        self.x += c * self.h[str(j)]

    def Output(self):
        return (self.x **2) ** 0.5

def main():
    tow = Tug_of_War()

    testdatastream = [1 for _ in range(1000)]
    iterations = 100

    x_axis = [i for i in range((iterations))]
    y_axis = []
    for _ in range(iterations):
        tow.Initialize()

        for p in range(len(testdatastream)):
            tow.Process(testdatastream[p], 1)

        y_axis.append(tow.Output())

    plt.plot(x_axis, y_axis, "ro")
    plt.show()

main()