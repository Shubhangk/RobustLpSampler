import random
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.stats import levy_stable

map_lp = {}
map_robust_lp = {}



class Tug_of_War:
    h = {}
    x = 0
    n = 0

    def Initialize(self, n):
        self.h = {}
        self.x = 0
        self.n = n
        for j in range(n):
            self.h[str(j)] = np.random.choice([-1, 1])

    def Process(self, j, c):
        #if self.h.get(str(j)) is None:
            #self.h[str(j)] = np.random.choice([-1, 1])

        self.x += c * self.h[str(j)]

    def Output(self):
        return self.x ** 2




debug = False

class HashFunction:
    DEFAULT = -1

    t = DEFAULT
    b = DEFAULT

    # table of hash functions into buckets
    hashFunctions_h = []
    # table for hash functions into range {-1, 1}
    hashFunctions_s = []

    def init(self, t, b):
        t = int(t)
        b = int(b)
        self.t = t
        self.b = b

        self.hashFunctions_h = [{} for _ in range(t)]
        self.hashFunctions_s = [{} for _ in range(t)]

    def getHash_h(self, index, key):
        #hashfunction = self.hashFunctions[index]

        if self.hashFunctions_h[index].get(str(key)) is None:
            sequence = [i for i in range(self.b)]
            self.hashFunctions_h[index][str(key)] = np.random.choice(sequence)

        return self.hashFunctions_h[index][str(key)]

    def getHash_s(self, index, key):
        #hashfunction_s = self.hashFunctions_s[index]

        if self.hashFunctions_s[index].get(str(key)) is None:
            sequence = [-1, 1]
            self.hashFunctions_s[index][str(key)] = np.random.choice(sequence)

        return self.hashFunctions_s[index][str(key)]

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
    p = DEFAULT

    scales = []

    def init(self, t, b, scales, p):
        t = int(t)
        b = int(b)
        self.scales = scales
        self.t = t
        self.b = b
        self.p = p

        # initialize hashfunctions into buckets
        self.H = HashFunction()
        self.H.init(t, b)

        # initialize frequencies to 0 for every hashed bucket
        for i in range(int(t)):
            self.frequencies.append([0 for _ in range(b)])


    def PrintStructure(self):
        print(self.H)
        print(self.frequencies)
        print(self.scales)

    def Add(self, q):
        for i in range(self.t):
            hi_OF_q = self.H.getHash_h(i, q)
            si_OF_q = self.H.getHash_s(i, q)

            # update the frequency of the appropriate bucket
            self.frequencies[i][hi_OF_q] += (si_OF_q / pow(self.scales[q], 1/float(self.p)))

        return

    def Estimate(self, q):
        estimates = []
        for i in range(self.t):
            hi_OF_q = self.H.getHash_h(i, q)
            si_OF_q = self.H.getHash_s(i, q)

            q_estimate = self.frequencies[i][hi_OF_q] * si_OF_q
            estimates.append(q_estimate)

        return int(np.median(estimates))



def p_stableDistribution(p):
    r = np.random.random()

    PI = math.pi
    theta = random.uniform(-PI / 2.0, PI / 2.0)

    # convenience notation
    Xl = math.sin(p * theta) / (math.cos(theta) ** (1 / float(p)))

    Xr = math.cos(theta * (1 - p)) / -math.log(r, math.e)
    Xr = Xr ** ((1-p)/p)

    X = Xl * Xr
    return X

def Abs_p_stableDistribution_median(p):
    r = np.random.random()

    PI = math.pi
    theta = random.uniform(-PI / 2.0, PI / 2.0)

    # convenience notation
    Xl = math.sin(p * theta) / (math.cos(theta) ** (1 / float(p)))

    Xr = math.cos(theta * (1 - p)) / -math.log(r, math.e)
    Xr = Xr ** ((1-p)/p)

    X = Xl * Xr
    return abs(X)

def median(p):
    return np.median([Abs_p_stableDistribution_median(p) for _ in range(100000)])


class LpEstimator:
    A = []
    r = -1
    p = -1
    y = []

    def initialize(self, n, p, eps):
        CONSTANT = 1000
        self.r = CONSTANT * (1 / float(eps))
        self.p = p

        # levy_alpha = 0.5
        # levy_beta = 1
        # levy_stable_distribution = levy_stable(levy_alpha, levy_beta)

        for i in range(int(self.r)):
            # levy_stable_samples = levy_stable_distribution.rvs(n)
            # self.A.append(levy_stable_samples)
            self.A.append([p_stableDistribution(p) for _ in range(n)])

        self.y = np.multiply(self.A, [0 for _ in range(n)])

        self.y = np.multiply(self.A, [0 for _ in range(n)])

    def update(self, i):
        for j in range(len(self.y)):
            self.y[j] += self.A[j][i]


    def output(self):
        temp = [abs(self.y[i]) for i in range(len(self.y))]
        median_y = np.median(temp)

        MEDIAN_Dp = median(self.p)

        return float(median_y) / abs(float(MEDIAN_Dp))

class HardcodedAMS_Estimator:
    t = []
    x = []
    n = 0
    p = 0

    def Initialize(self, t, p, n):
        self.p = p
        self.t = t
        self.n = n
        self.x = [0 for _ in range(n)]

    def Process(self, j):
        self.x[j] += 1

    def Output(self):
        temp = [pow(self.x[i] / pow(self.t[i], 1/float(self.p)), self.p) for i in range(self.n)]
        temp_sum = sum(temp)
        return pow(temp_sum, self.p)

class LpSampler:
    p = -1
    m = -1
    k = -1
    eps = -1
    B = -1
    l = -1
    t = []
    n = -1

    cs = None
    lp = None
    l2 = None


    def Initialize(self, n, p, eps):
        CONSTANT1 = 100
        CONSTANT2 = 100
        self.p = p
        self.eps = eps
        self.n = n

        self.k = 10 * math.ceil(1 / (abs(p-1)))
        self.m = CONSTANT1 * pow(eps, -max(0, p-1))
        self.B = pow(eps, 1-(1/float(p)))
        self.l = CONSTANT2 * math.log(n, 2)

        self.t = [np.random.random()  for _ in range(n)]
        self.cs = CountSketch()
        self.cs.init(t=self.l, b=(6 * self.m), scales=self.t, p=self.p)

        self.lp = LpEstimator()
        self.lp.initialize(n=n, p=p, eps=(pow(2, 0.5)-1))

        # self.l2 = AMS_Estimator_Proxy()
        # self.l2 = HardcodedAMS_Estimator()
        # self.l2.Initialize(t=self.t, p = self.p, n=self.n)
        self.l2 = Tug_of_War()
        self.l2.Initialize(self.n)

    def Process(self, j):
        self.cs.Add(j)
        self.lp.update(j)
        self.l2.Process(j, 1 / pow(self.t[j], 1/float(self.p)))

    def Recover(self):

        # 1. Find z* - get all estimates from count sketch
        temp_z_star = []
        z_star = []
        for i in range(self.n):
            z_star_i = self.cs.Estimate(i)
            z_star.append(z_star_i)
            temp_z_star.append(z_star_i)

        # 1. Find z^ - m sparse approximation of z*
        tempMax = float("-inf")
        maxIndex = -1
        z_cap = [0 for _ in range(len(z_star))]
        for i in range(int(self.m)):
            for j in range(len(z_star)):
                if temp_z_star[j] >= tempMax:
                    tempMax = temp_z_star[j]
                    maxIndex = j
            if tempMax == float("-inf"):
                break
            z_cap[maxIndex] = temp_z_star[maxIndex]
            temp_z_star[maxIndex] = float("-inf")
            tempMax = float("-inf")
            maxIndex = -1

        # 2. set epsilon to sqrt 2
        r = pow(2, 0.5) * self.lp.output()

        # 3. set epsilon to sqrt 2
        for i in range(len(z_cap)):
            self.l2.x -= z_cap[i] * self.l2.h[str(i)]
        l2_estimate = self.l2.Output()
        s = pow(2, 0.5) * l2_estimate


        # 4. find maximal |z_star| and corresponding index
        maximal_i = -1
        maximal_z_star = float("-inf")
        for a in range(len(z_star)):
            if abs(z_star[a]) > maximal_z_star:
                maximal_z_star = abs(z_star[a])
                maximal_i = a

        # 5. error check
        # if s > self.B * pow(self.m, 0.5) * r:
        #     # return failure
        #     print("fail1")
        #     return float("inf")
        # if maximal_z_star < pow(self.eps, -1/float(self.p)) * r:
        #     print("fail2")
        #     return float("inf")

        # 6. return (sample, approximation)
        return [maximal_i, maximal_z_star * pow(self.t[maximal_i], 1/float(self.p))]


def main():
    lp = LpSampler()

    epochs = 1
    test_vector = [0,1,2,3,4]
    testStreamsize = 100
    y_axis = [0 for _ in range(len(test_vector) + 1)]

    for e in range(epochs):
        print(str(e) + " / " + str(epochs) +" epochs complete.")
        lp.Initialize(n=5, p=0.5, eps=0.2)
        TestDataStream1 = [np.random.choice(test_vector) for _ in range(testStreamsize)]
        for tempvar in range(30):
            TestDataStream1[tempvar] = 1
        np.random.shuffle(TestDataStream1)

        for i in range(testStreamsize):
            lp.Process(TestDataStream1[i])

        res = lp.Recover()
        if res == float("inf"):
            y_axis[len(test_vector)] += 1
        else:
            index = res[0]
            y_axis[index] += 1

    test_vector.append("FAIL")
    plt.plot(test_vector, y_axis, "ro")
    plt.show()


class RobustLpSampler:

    lpsampler = None

    def Initailize(self, n, p, eps):
        self.lpsampler = LpSampler()
        self.lpsampler.Initialize(n, p, eps)

    def Process(self, j):
        # round the point first and send to lp sampler to process
        if j < 0 or j == int(j):
            j = int(j)
        else:
            j = int(j + 1)
        self.lpsampler.Process(j)

    def Output(self):
        return self.lpsampler.Recover()


#main()

def testUniform():
    n = 20
    p = 1.5
    eps = 0.01

    DATASET_SIZE = 100

    testvector = [i for i in range(n)]
    dataset = [np.random.choice(testvector) for i in range (DATASET_SIZE)]
    for tempvar in range(30):
        dataset[tempvar] = 1

    noisy_dataset = []
    for i in range(len(dataset)):
        point = dataset[i]

        # noise
        err = np.random.random()

        neighbor = point + err
        noisy_dataset.append(neighbor)

        #noisy_dataset.append(point)

    #np.random.shuffle(noisy_dataset)

    # testing

    lpsampler = LpSampler()
    robust_lpsampler = RobustLpSampler()

    epochs = 10
    x_axis = [i for i in range(n)]

    y_axis_lp = [0 for _ in range(n + 1)]
    y_axis_robust_lp = [0 for _ in range(n + 1)]

    for e in range (epochs):
        print(str(e) + " / " + str(epochs) + " epochs complete.")
        lpsampler.Initialize(n=n, p=p, eps=eps)
        robust_lpsampler.Initailize(n=n, p=p, eps=eps)

        print("initialized")
        for i in range(len(dataset)):
            lpsampler.Process(dataset[i])

        print("lpsampler processing done")
        for j in range(len(noisy_dataset)):
            robust_lpsampler.Process(noisy_dataset[j])

        print("robust lp sampler processing done")
        res_lp = lpsampler.Recover()
        if res_lp == float("inf"):
            y_axis_lp[n] += 1
        else:
            index = res_lp[0]
            y_axis_lp[index] += 1
        print("recovered lp sample")

        res_robust_lp = robust_lpsampler.Output()
        if res_robust_lp == float("inf"):
            y_axis_robust_lp[n] += 1
        else:
            index = res_robust_lp[0]
            y_axis_robust_lp[index] += 1
        print("recovered robust lp sample")

    x_axis.append(len(x_axis))
    print(x_axis)
    print(y_axis_lp)
    print(y_axis_robust_lp)
    plt.plot(x_axis, y_axis_lp, "red", x_axis, y_axis_robust_lp, "blue")
    plt.ylabel("Output Frequency")
    plt.xlabel("Sample")

    #plt.savefig("n_20_stream_100_iterations_20", bbox_inches='tight')
    plt.show()


#testUniform()
def Tester(p, stream, epochs):
    n = 20
    eps = 0.01

    DATASET_SIZE = 500

    vector = [i for i in range(n)]

    dataset = stream
    noisy_dataset = []
    for i in range(len(dataset)):
        point = dataset[i]

        # noise
        err = np.random.random()

        neighbor = point + err
        noisy_dataset.append(neighbor)

    lpsampler = LpSampler()
    robust_lpsampler = RobustLpSampler()

    x_axis = [i for i in range(n)]

    y_axis_lp = [0 for _ in range(n + 1)]
    y_axis_robust_lp = [0 for _ in range(n + 1)]

    for e in range(epochs):
        if e % 1 == 0:
            print(str(e) + " / " + str(epochs) + " epochs complete.")

        lpsampler.Initialize(n=n, p=p, eps=eps)
        robust_lpsampler.Initailize(n=n, p=p, eps=eps)

        for i in range(len(dataset)):
            lpsampler.Process(dataset[i])

        for j in range(len(noisy_dataset)):
            robust_lpsampler.Process(noisy_dataset[j])

        res_lp = lpsampler.Recover()
        if res_lp == float("inf"):
            y_axis_lp[n] += 1
        else:
            index = res_lp[0]
            y_axis_lp[index] += 1

        res_robust_lp = robust_lpsampler.Output()
        if res_robust_lp == float("inf"):
            y_axis_robust_lp[n] += 1
        else:
            index = res_robust_lp[0]
            y_axis_robust_lp[index] += 1

    x_axis.append(len(x_axis))
    print(x_axis)
    print(y_axis_lp)
    print(y_axis_robust_lp)
    print(str(n))
    print(str(eps))
    print(str(DATASET_SIZE))
    print(str(p))
    print("---------")


def TestUniform500():
    n = 20
    p = 1.5
    epochs = 1000
    DATASET_SIZE = 500

    testvector = [i for i in range(n)]
    dataset = [np.random.choice(testvector) for i in range (DATASET_SIZE)]

    Tester(1.5, dataset, epochs)

    Tester(0.5, dataset, epochs)

def TestSingleBias():
    n = 20
    p = 1.5
    epochs = 1000
    DATASET_SIZE = 500
    testvector = [i for i in range(n)]

    bias = 3
    bias_size = 100

    dataset = [np.random.choice(testvector) for i in range(DATASET_SIZE)]
    for tempvar in range(bias_size):
        dataset[tempvar] = bias
    np.random.shuffle(dataset)

    Tester(1.5, dataset, epochs)

    Tester(0.5, dataset, epochs)

def TestMultiBias():
    n = 20
    p = 1.5
    epochs = 1000
    DATASET_SIZE = 500
    testvector = [i for i in range(n)]

    bias2 = 15
    bias1 = 2
    bias_size = 80



    dataset = [np.random.choice(testvector) for i in range(DATASET_SIZE)]

    for tempvar in range(bias_size):
        dataset[tempvar] = bias1
    for tempvar in range(bias_size):
        dataset[tempvar+bias_size] = bias2
    np.random.shuffle(dataset)

    Tester(1.5, dataset, epochs)

    Tester(0.5, dataset, epochs)

def HelloWorld():
    m = 10
    n = 20
    p = 1.5
    epochs = 10
    DATASET_SIZE = 100
    eps = 0.01

    testvector = [i for i in range(m)]

    dataset = [np.random.choice(testvector) for _ in range(DATASET_SIZE)]
    dataset = [2 * dataset[i] for i in range(DATASET_SIZE)]
    noisy_dataset = []

    for tempvar in range(25):
        dataset[tempvar] = 2

    for i in range(len(dataset)):
        point = dataset[i]

        # noise
        err = np.random.random()

        neighbor = point + err * np.random.choice([-1,1])
        noisy_dataset.append(neighbor)


    lpsampler = LpSampler()
    robust_lpsampler = RobustLpSampler()
    n += 1
    x_axis = [i for i in range(n)]

    y_axis_lp = [0 for _ in range(n + 1)]
    y_axis_robust_lp = [0 for _ in range(n + 1)]

    for e in range(epochs):
        if e % 1 == 0:
            print(str(e) + " / " + str(epochs) + " epochs complete.")

        lpsampler.Initialize(n=n, p=p, eps=eps)
        robust_lpsampler.Initailize(n=n, p=p, eps=eps)

        for i in range(len(dataset)):
            lpsampler.Process(dataset[i])

        for j in range(len(noisy_dataset)):
            robust_lpsampler.Process(noisy_dataset[j])

        res_lp = lpsampler.Recover()
        if res_lp == float("inf"):
            y_axis_lp[n] += 1
        else:
            index = res_lp[0]
            y_axis_lp[index] += 1

        res_robust_lp = robust_lpsampler.Output()
        if res_robust_lp == float("inf"):
            y_axis_robust_lp[n] += 1
        else:
            index = res_robust_lp[0]
            y_axis_robust_lp[index] += 1

    x_axis.append(len(x_axis))
    print(x_axis)
    print(y_axis_lp)
    print(y_axis_robust_lp)
    print(str(n))
    print(str(eps))
    print(str(DATASET_SIZE))
    print(str(p))
    print("---------")


def driver(m, p, epochs, eps, stream_size, stream):
    n = 2 * m
    testvector = [i for i in range(m)]

    noisy_dataset = []

    for i in range(len(stream)):
        point = stream[i]

        # noise
        err = np.random.random()

        neighbor = point + err * np.random.choice([-1, 1])
        noisy_dataset.append(neighbor)

    lpsampler = LpSampler()
    robust_lpsampler = RobustLpSampler()
    n += 1
    x_axis = [i for i in range(n)]

    y_axis_lp = [0 for _ in range(n + 1)]
    y_axis_robust_lp = [0 for _ in range(n + 1)]

    for e in range(epochs):
        if e % 10 == 0:
            print(str(e) + " / " + str(epochs) + " epochs complete.")
            print(y_axis_lp)
            print(y_axis_robust_lp)
            print("--------")

        lpsampler.Initialize(n=n, p=p, eps=eps)
        robust_lpsampler.Initailize(n=n, p=p, eps=eps)

        for i in range(len(stream)):
            lpsampler.Process(stream[i])

        for j in range(len(noisy_dataset)):
            robust_lpsampler.Process(noisy_dataset[j])

        res_lp = lpsampler.Recover()
        if res_lp == float("inf"):
            y_axis_lp[n] += 1
        else:
            index = res_lp[0]
            y_axis_lp[index] += 1

        res_robust_lp = robust_lpsampler.Output()
        if res_robust_lp == float("inf"):
            y_axis_robust_lp[n] += 1
        else:
            index = res_robust_lp[0]
            y_axis_robust_lp[index] += 1

    x_axis.append(len(x_axis))
    print(x_axis)
    print(y_axis_lp)
    print(y_axis_robust_lp)
    print(str(n))
    print(str(eps))
    print(str(stream_size))
    print(str(p))
    print("---------")

def generateUniform(length, m):
    a = [i for i in range(m)]
    dataset = [2 * np.random.choice(a) for _ in range(length)]
    return dataset

GLOBAL_UNIFORM_DATASET = [30, 30, 22, 10, 36, 2, 34, 26, 34, 20, 38, 16, 16, 22, 28, 6, 0, 8, 0, 6, 28, 30, 38, 22, 36, 20, 30, 32, 28, 4, 22, 38, 30, 0, 8, 30, 34, 36, 20, 34, 12, 14, 2, 34, 20, 2, 32, 4, 14, 14, 34, 30, 24, 8, 38, 34, 32, 8, 12, 10, 2, 0, 10, 34, 36, 6, 18, 36, 36, 30, 10, 38, 30, 32, 4, 26, 22, 36, 6, 8, 4, 26, 2, 34, 18, 34, 12, 28, 8, 6, 8, 8, 10, 26, 6, 10, 4, 8, 28, 30]
GLOBAL_BIASED_DATASET = [2, 18, 38, 2, 16, 10, 30, 20, 36, 24, 2, 30, 20, 20, 10, 2, 18, 2, 22, 0, 28, 10, 36, 26, 2, 4, 2, 38, 28, 8, 24, 16, 30, 20, 34, 28, 28, 32, 2, 18, 16, 16, 36, 0, 14, 0, 16, 26, 4, 16, 32, 12, 22, 34, 20, 16, 38, 22, 32, 0, 32, 36, 8, 2, 14, 14, 2, 6, 0, 2, 34, 22, 2, 24, 32, 10, 2, 6, 2, 26, 24, 30, 2, 0, 10, 2, 2, 34, 2, 2, 38, 28, 24, 0, 20, 2, 28, 2, 2, 8]

if __name__ == '__main__':
    import sys

    # parse arguments
    m = int(sys.argv[1])
    p = float(sys.argv[2])
    epochs = int(sys.argv[3])
    eps = float(sys.argv[4])
    stream_size = int(sys.argv[5])

    stream_option = int(sys.argv[6])

    stream = []
    if stream_option == 1:
        stream = GLOBAL_UNIFORM_DATASET
    else:
        stream = GLOBAL_BIASED_DATASET

    # call driver
    try:
        driver(m, p, epochs, eps, stream_size, stream)
    except:
        print(str(m))
        print(str(eps))
        print(str(stream_size))
        print(str(p))
        print("---------")


# deprecated tests. 
#TestMultiBias()
#TestUniform500()
#TestSingleBias()
