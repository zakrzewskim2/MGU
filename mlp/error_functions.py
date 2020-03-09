import numpy as np


def __delta(y_predicted, y):
    result = np.where(y < y_predicted,
                      1,
                      np.where(y > y_predicted, 0, -1))
    return result


def mean_squared(y_predicted, y, derivative=False):
    if derivative:
        return (2 * (y_predicted - y) / y.shape[1])
    else:
        return ((y_predicted - y)**2).mean()


def mean(y_predicted, y, derivative=False):
    if derivative:
        return __delta(y_predicted, y)
    else:
        return np.abs(y_predicted - y).mean()


def max_error(y_predicted, y, derivative=False):
    if derivative:
        indices = np.where(np.max(np.abs(y_predicted - y), axis=1) == np.abs(y_predicted - y).T)
        result = np.where(indices,
                          2 * (y_predicted[indices] >
                               y[indices]) - 1,  # to get -1/1
                          0)
        return result
    else:
        return np.max(np.abs(y_predicted - y), axis=1).mean()


def cross_entropy(y_predicted, y, derivative=False):
    if derivative:
        return -y / y_predicted
    else:
        return np.sum(-y * np.log(y_predicted))