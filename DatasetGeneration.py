import numpy as np

a = [i for i in range(20)]
dataset = [2 * np.random.choice(a) for _ in range(100)]
for tempvar in range(20):
    dataset[tempvar] = 2
np.random.shuffle(dataset)

print(dataset)
