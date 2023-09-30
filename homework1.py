import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
import numpy as np
import torch
import torch.nn as nn

from layer import Linear, Sigmoid, Softmax, ReLu, CrossEntropyLoss, Tanh

def load_data():
    X, y = make_moons(n_samples=10000, noise=0.05, random_state=0)
    return X, y

def plot_data(X, y):
    plt.scatter(X[:, 0], X[:, 1], c=y)
    plt.show()

if __name__ == '__main__':
    X, y = load_data()
    plot_data(X, y)
