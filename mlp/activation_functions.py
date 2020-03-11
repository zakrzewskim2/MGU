import numpy as np


def softmax(x, derivative=False):
    x = x - np.max(x)
    if not derivative:
        e_x = np.exp(x)
        return e_x / e_x.sum()
    else:
        diag_3d = np.zeros((x.shape[1], x.shape[0], x.shape[1]))
        np.einsum('iji->ji', diag_3d)[...] = np.ones((x.shape[0], x.shape[1]))
        diag_3d = diag_3d * (softmax(x)*(1-softmax(x)))
        some_2d_matrix = softmax(x)[:, :, None]
        we_are_getting_there_matrix = np.transpose(some_2d_matrix, (0, 2, 1))
        almost_done_matrix = -we_are_getting_there_matrix*some_2d_matrix
        so_close_matrix = np.transpose(almost_done_matrix, (1, 0, 2))
        np.einsum(
            'iji->ji', so_close_matrix)[...] = np.zeros((x.shape[0], x.shape[1]))
        result = so_close_matrix+diag_3d
        return result


def __to_3d_derivative(x, base_derivative):
    diag_3d = np.zeros((x.shape[1], x.shape[0], x.shape[1]))
    np.einsum('iji->ji', diag_3d)[...] = np.ones((x.shape[0], x.shape[1]))
    if type(base_derivative) == int:
        return base_derivative * diag_3d
    return diag_3d * base_derivative[None, :, :]


def sigmoid(x, derivative=False):
    if not derivative:
        return 1 / (1 + np.exp(-x))
    else:
        base_derivative = sigmoid(x) * (1 - sigmoid(x))
        return __to_3d_derivative(x, base_derivative)


def tanh(x, derivative=False):
    if not derivative:
        return 0.5 * (np.tanh(x) + 1)
    else:
        base_derivative = 1 / (np.cosh(2 * x) + 1)
        return __to_3d_derivative(x, base_derivative)


def linear(x, derivative=False):
    if not derivative:
        return x
    else:
        base_derivative = 1
        return __to_3d_derivative(x, base_derivative)
