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
        return 1 #x * (1 - x)
    else:
        return x #1 / (1 + np.exp(-x))
#%%
import numpy as np

# inputs - each row is an input to MLP
X = np.array([  
    [0, 0, 1],
    [0, 1, 1],
    [1, 1, 1],
])

# N - number of input vectors
N = X.shape[0]

# outputs - each row is an expected output for the corresponding input
y = np.array([  
    [1, 0],
    [0, 1],
    [0, 0],
])

# number of layers (all of them!)
num_layers = 4

# number of nodes in each layer
layer_lengths = np.array([X.shape[1], 4, 5, y.shape[1]])
if layer_lengths.shape[0] != num_layers:
    print("Error! Number of layers undefined!")

#%%
np.random.seed(1)
weigth_matrices = []
for i in range(0, num_layers-1):
    # Initial weights from [-1,1], with first row as bias
    weigth_matrices.append(2*np.random.random((layer_lengths[i] + 1, layer_lengths[i+1])) - 1)

# number of iterations
num_iterations = 10000
# learning rate
alpha = 0.1

for i in range(num_iterations):
    # output of each layer
    # outputs[0] - inputs with ones in first column (input for bias)
    # outputs[1] - output of first hidden layer, with ones in first column
    # ...
    # outputs[-1] - output of last layer (without ones)
    outputs = []
    outputs.append(np.hstack((np.ones((X.shape[0], 1)), X)))
    for i in range(0, num_layers-2):
        outputs.append(np.hstack((np.ones((X.shape[0], 1)), activation(np.dot(outputs[i], weigth_matrices[i])))))
    outputs.append(np.dot(outputs[-1], weigth_matrices[-1]))

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
            np.dot(errors[i], weigth_matrices[-i-1].T[:, 1:]))

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
        weigth_matrices[i] += -alpha * gradients[i]

#%%
result = []
result.append(np.hstack((np.ones((X.shape[0], 1)), X)))
for i in range(0, num_layers-2):
    result.append(np.hstack((np.ones((X.shape[0], 1)), activation(np.dot(result[i], weigth_matrices[i])))))
result.append(np.dot(result[-1], weigth_matrices[-1]))

with np.printoptions(precision=3, suppress=True):
    print("Expected output:\n", y)
    print("Output After Training:\n", result[-1])
# %%
