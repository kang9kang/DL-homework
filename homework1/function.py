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


def batch_norm(x, running_mean, running_var, weight, bias, training, momentum=0.1, eps=1e-5):
    if training:
        mean = np.mean(x, axis=0)
        var = np.var(x, axis=0)
        x_hat = (x - mean) / np.sqrt(var + eps)
        running_mean = momentum * mean + (1 - momentum) * running_mean
        running_var = momentum * var + (1 - momentum) * running_var
    else:
        x_hat = (x - running_mean) / np.sqrt(running_var + eps)
    return x_hat * weight + bias