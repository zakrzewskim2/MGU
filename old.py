#%%
def activation(x, derivative=False):
    if (derivative == True):
        return x * (1 - x)
    else:
        return 1 / (1 + np.exp(-x))

def error_function(y, y_exp, derivative = False):
    if (derivative == True):
        return 2 * (y - y_exp) / y_exp.shape[1]
    else:
        return ((y - y_exp)**2).mean()

def out_activation(x, derivative=False):
    if (derivative == True):
        return 1 
    else:
        return x 
#%%
import numpy as np
import pandas as pd

train = pd.read_csv("data\classification\data.simple.train.100.csv")
test = pd.read_csv("data\classification\data.simple.test.100.csv")
X_train = train.iloc[:,:-1]
y_train = train["cls"]
X_test = np.array(test.iloc[:,:-1])
y_test = np.array([test["cls"]]).T
# inputs - each row is an input to MLP
X = np.array(X_train)
# outputs - each row is an expected output for the corresponding input
y = np.array([y_train]).T
# N - number of input vectors
N = X.shape[0]

# number of layers (all of them!)
num_layers = 5

# number of nodes in each layer
layer_lengths = np.array([X.shape[1], 5, 5, 3, y.shape[1]])
if layer_lengths.shape[0] != num_layers:
    print("Error! Number of layers undefined!")

#%%
np.random.seed(1)
weight_matrices = []
for i in range(0, num_layers-1):
    # Initial weights from [-1,1], with first row as bias
    weight_matrices.append(2*np.random.random((layer_lengths[i] + 1, layer_lengths[i+1])) - 1)

# number of iterations
num_iterations = 100000
# learning rate
alpha = 0.1

for j in range(num_iterations):
    # output of each layer
    # outputs[0] - inputs with ones in first column (input for bias)
    # outputs[1] - output of first hidden layer, with ones in first column
    # ...
    # outputs[-1] - output of last layer (without ones)
    outputs = []
    outputs.append(np.hstack((np.ones((X.shape[0], 1)), X)))
    for i in range(0, num_layers-2):
        outputs.append(np.hstack((np.ones((X.shape[0], 1)), activation(np.dot(outputs[i], weight_matrices[i])))))
    outputs.append(np.dot(outputs[-1], weight_matrices[-1]))

    # error term for each layer
    # errors[0] - error for last layer
    # errors[1] - error for last hidden layer
    # ...
    # errors[-1] - error for first hidden layer
    errors = []
    errors.append(out_activation(outputs[-1], True) * \
        error_function(out_activation(outputs[-1]), y, True))
    for i in range(0, num_layers-2):
        errors.append(activation(outputs[-i-2][:, 1:], True) * \
            np.dot(errors[i], weight_matrices[-i-1].T[:, 1:]))

    # gradient for each matrix of weights
    # gradient[0] - gradient with respect to weights between input layer and first hidden layer
    # gradient[1] - gradient with respect to weights between first and second hidden layer
    # ...
    # gradient[-1] - gradient with respect to weights between last hidden layer and output layer
    gradients = []
    for i in range(0, num_layers-1):
        gradients.append(np.dot(outputs[i].T, errors[-i-1]) / N)

    # Adjusting weights
    for i in range(0, num_layers-1):
        weight_matrices[i] += -alpha * gradients[i]
    
    if j % 1000 == 0:
        print("Error:", error_function(y, outputs[-1]))

#%%
result = []
result.append(np.hstack((np.ones((X_test.shape[0], 1)), X_test)))
for i in range(0, num_layers-2):
    result.append(np.hstack((np.ones((X_test.shape[0], 1)), activation(np.dot(result[i], weight_matrices[i])))))
result.append(np.dot(result[-1], weight_matrices[-1]))

with np.printoptions(precision=3, suppress=True):
    print("Expected output:\n", y_test.T)
    print("Output After Training:\n", result[-1].T)
    print("Error:\n", error_function(y_test, result[-1]))
# %%