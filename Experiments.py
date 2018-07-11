import numpy as np

def Tester(p, stream):
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

    epochs = 10
    x_axis = [i for i in range(n)]

    y_axis_lp = [0 for _ in range(n + 1)]
    y_axis_robust_lp = [0 for _ in range(n + 1)]

    for e in range(epochs):
        if e % 50 == 0:
            print(str(e) + " / " + str(epochs) + " epochs complete.")

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


n = 20
p = 1.5
DATASET_SIZE = 100

testvector = [i for i in range(n)]
dataset = [np.random.choice(testvector) for i in range (DATASET_SIZE)]

Tester(1.5, dataset)