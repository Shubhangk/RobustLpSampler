import numpy as np

def test1():
    # generate data
    dataset = [np.random.random() for _ in range(50)]

    # get minimum pairwise distance
    min_dist = float("inf")
    for i in range(len(dataset)):
        for j in range(len(dataset)):
            min_dist = min(min_dist, abs(dataset[i] - dataset[j]))

    # scale the dataset so that min distance is 1
    dataset = [(dataset[i] / min_dist) for i in range(len(dataset))]

    mapper = [dataset[index] for index in range(len(dataset))]
    mapper.sort()
    global map_lp
    map_lp = {}
    for index in range(len(mapper)):
        map_lp[str(mapper[index])] = index

    noisy_dataset = []
    for i in range(len(dataset)):
        point = dataset[i]

        # number of near neighbors of dataset[i]
        k_i = np.random.choice([j for j in range(100)])

        # add k_i points
        for _ in range(k_i):
            # # scale - not
            # z = np.random.random()

            # noise
            delta = np.random.random()
            delta = delta / 2.0

            neighbor = dataset[i] + delta
            noisy_dataset.append(neighbor)

    noisy_dataset.append(i)

    alpha = 0.5
    binned_stream = []
    for i in range(len(noisy_dataset)):
        temp = noisy_dataset[i] - int(noisy_dataset[i])
        binned_point = 0
        if temp > alpha:
            binned_point = int(noisy_dataset[i]) + 2 * alpha
        else:
            binned_point = int(noisy_dataset[i]) + alpha

        binned_stream.append(binned_point)

    mapper_robust = [noisy_dataset[i] for i in range(len(noisy_dataset))]
    mapper_robust.sort()

    global map_robust_lp
    map_robust_lp = {}
    for index in range(len(mapper_robust)):
        map[str(mapper_robust[index])] = index

import matplotlib.pyplot as plt

x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
y1 = [3, 0, 3, 3, 2, 2, 2, 2, 1, 2, 5, 2, 3, 1, 3, 3, 1, 5, 4, 0, 13]
y2 = [2, 3, 3, 1, 3, 1, 3, 6, 3, 2, 5, 2, 2, 3, 2, 1, 2, 6, 1, 2, 7]

a = [1,2,3,4,5]
b = [3,4,5,6,7]
plt.plot(x, y1, "red",  x, y2, "blue")
plt.show()
