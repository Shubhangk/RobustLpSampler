import random
import numpy as np
import matplotlib.pyplot as plt

debug = False



class HashFunction:
    DEFAULT = -1

    t = DEFAULT
    b = DEFAULT

    hashFunctions = []
    hashFunctions_s = []

    def init(self, t, b):
        self.t = t
        self.b = b

        self.hashFunctions = [{} for _ in range(t)]
        self.hashFunctions_s = [{} for _ in range(t)]

    def getHash_h(self, index, key):
        hashfunction = self.hashFunctions[index]

        if hashfunction.get(str(key)) is None:
            sequence = [i for i in range(self.b)]
            hashfunction[str(key)] = np.random.choice(sequence)

        return hashfunction[str(key)]

    def getHash_s(self, index, key):
        hashfunction_s = self.hashFunctions_s[index]

        if hashfunction_s.get(str(key)) is None:
            sequence = [-1, 1]
            hashfunction_s[str(key)] = np.random.choice(sequence)

        return hashfunction_s[str(key)]

class CountSketch:
    DEFAULT = -1

    # Hash function object
    H = None
    # frequency estimates
    frequencies = []
    # scales for the co-ordinates
    scales = []
    # hyper-parameters
    t = DEFAULT
    b = DEFAULT

    def init(self, t, b):
        self.t = t
        self.b = b
        for i in range(t):
            # initialize hashfunctions into buckets
            self.H = HashFunction()
            self.H.init(t, b)

            # initialize frequencies to 0
            self.frequencies.append([0 for _ in range(b)])

        #self.scales = scales
        if debug:
            self.PrintStructure()
        return

    def PrintStructure(self):
        print(self.H)
        print(self.frequencies)
        print(self.scales)

    def Add(self, q):
        for i in range(self.t):
            hi_OF_q = self.H.getHash_h(i, q)
            si_OF_q = self.H.getHash_s(i, q)

            self.frequencies[i][hi_OF_q] += si_OF_q
        return

    def Estimate(self, q, scales):
        estimates = []
        for i in range(self.t):
            hi_OF_q = self.H.getHash_h(i, q)
            si_OF_q = self.H.getHash_s(i, q)

            q_estimate = self.frequencies[i][hi_OF_q] * si_OF_q
            q_estimate = q_estimate / float(scales[q])
            estimates.append(q_estimate)
        return int(np.median(estimates))

def main():
    n = 20

    test_vector = [i for i in range(n)]
    scales = [1 for i in range(n)]

    test_datastream = np.random.choice(test_vector, 10000)
    cs = CountSketch()
    cs.init(10, 10)
    for i in range(len(test_datastream)):
        point = test_datastream[i]
        cs.Add(point)

    estimates = [0 for _ in range(n)]
    for i in range(len(test_vector)):
        estimates[i] = cs.Estimate(i, scales)

    actual_frequencies = [0 for _ in range(n)]
    for i in range(len(test_datastream)):
        actual_frequencies[test_datastream[i]] += 1

    print(test_vector)
    print(actual_frequencies)
    print(estimates)

    diff = []
    for j in range(n):
        diff.append(abs(actual_frequencies[j] - estimates[j]))

    #plt.plot(test_vector, actual_frequencies, 'ro', test_vector, estimates, 'bs')
    plt.plot(test_vector, diff, test_vector, actual_frequencies, "ro")
    plt.show()

main()
