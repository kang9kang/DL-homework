from sklearn.datasets import make_moons
import matplotlib.pyplot as plt
import numpy as np


class DataSet:
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __getitem__(self, index):
        return self.X[index], self.y[index]

    def __len__(self):
        return len(self.X)


class HalfMoonDataSet(DataSet):
    def __init__(self, n_samples=1000, noise=0.05, random_state=0):
        X, y = make_moons(n_samples=n_samples, noise=noise, random_state=random_state)
        y = np.eye(2)[y]
        super(HalfMoonDataSet, self).__init__(X, y)

    def plot(self):
        plt.scatter(self.X[:, 0], self.X[:, 1], c=self.y)
        plt.show()


class DataLoader:
    def __init__(self, dataset, batch_size=1, shuffle=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_samples = len(dataset)
        self.num_batches = int(
            (self.num_samples + self.batch_size - 1) / self.batch_size
        )
        self.index = 0
        self.indices = list(range(self.num_samples))
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __iter__(self):
        return self

    def __next__(self):
        if self.index >= self.num_samples:
            raise StopIteration
        indices = self.indices[self.index : self.index + self.batch_size]
        batch = [self.dataset[i] for i in indices]
        X = np.array([x for x, _ in batch])
        y = np.array([y for _, y in batch])
        self.index += self.batch_size
        return X, y
