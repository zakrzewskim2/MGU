# %%
import joblib
import pandas as pd
import numpy as np
from params import Config
activation = Config.activation_function
out_activation = Config.activation_function
error_function = Config.error_function

# %%
train = pd.read_csv("data\classification\data.simple.train.100.csv")
test = pd.read_csv("data\classification\data.simple.test.100.csv")
X_train, y_train = train.iloc[:, :-1], train["cls"]
X_test, y_test = np.array(test.iloc[:, :-1]), np.array([test["cls"]]).T

# inputs - each row is an input to MLP
X = X_train.values

# outputs - each row is an expected output for the corresponding input
def one_hot_encode(y, min_class, max_class):
    return np.identity(max_class - min_class + 1)[y - min_class, :]

y = y_train.values
y = one_hot_encode(y, min_class = 1, max_class = 2)
y_test = one_hot_encode(y_test, min_class = 1, max_class = 2)

# N - number of input vectors
N = X.shape[0]

# number of layers (all of them!)
num_layers = len(Config.hidden_layers) + 2

# number of nodes in each layer
layer_lengths = np.array([X.shape[1]] + Config.hidden_layers + [y.shape[1]])
if layer_lengths.shape[0] != num_layers:
    print("Error! Number of layers undefined!")

# %%
np.random.seed(1)
weight_matrices = []
for i in range(0, num_layers-1):
    nrows = layer_lengths[i] + 1
    ncols = layer_lengths[i + 1]

    # Initial weights from [-1,1], with first row as bias
    weight_matrices.append(2 * np.random.random((nrows, ncols)) - 1)

# number of iterations
num_iterations = Config.num_iterations
# learning rate
alpha = Config.eta

def append_ones_column(X):
    # X = [[2, 3], [4, 5]] -> X = [[1, 2, 3], [1, 4, 5]]
    return np.insert(X, 0, 1, axis=1)

def layer_output(inputs, weights, activate=True):
    base_output = np.dot(inputs, weights)
    if activate:
        return append_ones_column(activation(base_output))
    else:
        return append_ones_column(base_output)

weight_history = [np.array([np.copy(x) for x in weight_matrices])]
for j in range(num_iterations):
    # output of each layer
    # outputs[0] - inputs with ones in first column (input for bias)
    # outputs[1] - output of first hidden layer, with ones in first column
    # ...
    # outputs[-1] - output of last layer (without ones)

    outputs = []
    not_activated_outputs = []

    outputs.append(append_ones_column(X))
    not_activated_outputs.append(outputs[0])

    for i in range(0, num_layers-2):
        activated_output = layer_output(outputs[i], weight_matrices[i], \
            activate=True)
        not_activated_output = layer_output(outputs[i], weight_matrices[i], \
            activate=False)
        outputs.append(activated_output)
        not_activated_outputs.append(not_activated_output)

    outputs.append(np.dot(outputs[-1], weight_matrices[-1]))
    not_activated_outputs.append(outputs[-1])

    # error term for each layer
    # errors[0] - error for last layer
    # errors[1] - error for last hidden layer
    # ...
    # errors[-1] - error for first hidden layer
    errors = []
    err = (np.transpose(error_function(out_activation(outputs[-1]), y, True)[None, :, :], (0, 2, 1)) *
           np.transpose(out_activation(outputs[-1], True), (0, 2, 1))).sum(axis=1)
    errors.append(err.T)
    for i in range(0, num_layers-2):
        errors.append((np.dot(weight_matrices[-i-1][1:, :], errors[i].T)[None, :, :] *
                       np.transpose(activation(not_activated_outputs[-i-2][:, 1:], True), (0, 2, 1))).sum(axis=1).T)

    # gradient for each matrix of weights
    # gradient[0] - gradient with respect to weights between input layer and first hidden layer
    # gradient[1] - gradient with respect to weights between first and second hidden layer
    # ...
    # gradient[-1] - gradient with respect to weights between last hidden layer and output layer
    gradients = []
    for i in range(0, num_layers-1):
        gradient = np.dot(outputs[i].T, errors[-i-1])/N
        gradients.append(gradient)

    # Adjusting weights
    for i in range(0, num_layers-1):
        weight_matrices[i] += -alpha * gradients[i]

    if j % 1000 == 0:
        print("Error:", error_function(out_activation(outputs[-1]), y))
    
    weight_history.append(np.array([np.copy(x) for x in weight_matrices]))

training_data = {
    'weight_history': weight_history, 
    'config': Config(),
    'layer_lengths': layer_lengths
}
joblib.dump(training_data, 'training_data.joblib')

# %%
result = []
result.append(append_ones_column(X_test))
for i in range(0, num_layers-2):
    activated_result = layer_output(result[i], weight_matrices[i], \
        activate=True)
    result.append(activated_result)
result.append(np.dot(result[-1], weight_matrices[-1]))

with np.printoptions(precision=3, suppress=True):
    print("Expected output:\n", y_test.T)
    print("Output After Training:\n", out_activation(result[-1]))
    print("Error:\n", error_function(out_activation(result[-1]), y_test))

# %%