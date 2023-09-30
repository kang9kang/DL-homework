import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def softmax(x):
    x = x - np.max(x)
    expX = np.exp(x)
    return expX / np.sum(expX, axis=-1, keepdims=True)


def relu(x):
    return np.maximum(x, 0)


def cross_entropy(y, t):
    return -np.sum(t * np.log(y)) / len(y)


def tanh(x):
    return np.tanh(x)


def linear(x, W, b):
    return np.dot(x, W) + b
