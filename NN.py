#%%
import params
from params import ActivationFunction
def softmax(x):
    denominators = np.sum(np.exp(x), axis=1)
    return np.exp(x) / denominators[:, None]
def softmax_derivative(x):
    diag_3d = np.zeros((x.shape[1], x.shape[0], x.shape[1]))
    np.einsum('iji->ji', diag_3d)[...] = np.ones((x.shape[0], x.shape[1]))
    diag_3d = diag_3d * (softmax(x)*(1-softmax(x)))
    some_2d_matrix = softmax(x)[:,:,None]
    we_are_getting_there_matrix = np.transpose(some_2d_matrix, (0,2,1))
    almost_done_matrix = -we_are_getting_there_matrix*some_2d_matrix
    so_close_matrix = np.transpose(almost_done_matrix, (1,0,2))
    np.einsum('iji->ji', so_close_matrix)[...] = np.zeros((x.shape[0], x.shape[1]))
    result = so_close_matrix+diag_3d
    return result
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
def sigmoid_derivative(x):
    return x * (1 - x) #sigmoid(x) * (1 - sigmoid(x))
def activation_function(x, type, derivative=False):
    if derivative:
        if type == ActivationFunction.SOFTMAX:
            return softmax_derivative(x)
        diag_3d = np.zeros((x.shape[1], x.shape[0], x.shape[1]))
        np.einsum('iji->ji', diag_3d)[...] = np.ones((x.shape[0], x.shape[1]))
        return {
            ActivationFunction.SIGMOID: diag_3d*sigmoid_derivative(x)[None,:,:],
            ActivationFunction.TANH: diag_3d*(1 / (np.cosh(2 * x) + 1)),
            ActivationFunction.LINEAR: diag_3d
        }.get(type, diag_3d)

    else:
        return {
            ActivationFunction.SIGMOID: sigmoid(x),
            ActivationFunction.TANH: 0.5 * (np.tanh(x)+1),
            ActivationFunction.SOFTMAX: softmax(x),
            ActivationFunction.LINEAR: x
        }.get(type, x)

def error_function(y, y_exp, derivative = False):
    if (derivative == True):
        return (2 * (y - y_exp) / y_exp.shape[1])
    else:
        return ((y - y_exp)**2).mean()

def activation(x, derivative=False):
    return activation_function(x,ActivationFunction.TANH, derivative)

def out_activation(x, derivative=False):
    return activation_function(x, ActivationFunction.LINEAR, derivative)
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
X = np.array(X_train)#[0:4,:]
# outputs - each row is an expected output for the corresponding input
y = np.array([y_train]).T#[0:4,:]
# yy = np.zeros((100,2))
# yy.T[0] = y.T[0]
# yy.T[1] = y.T[0]
# for i in range(100):
#     if yy[i][0]==1:
#         yy[i][0]=1
#         yy[i][1]=0
#     else:
#         yy[i][0]=0
#         yy[i][1]=1
# y = yy
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
num_iterations = 10000
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
        error_function(out_activation(outputs[-1]), out_activation(y), True))
    for i in range(0, num_layers-2):
        errors.append(activation(outputs[-i-2][:, 1:], True) * \
            np.sum(np.dot(errors[i], weight_matrices[-i-1].T[:, 1:]), axis=0))

    # gradient for each matrix of weights
    # gradient[0] - gradient with respect to weights between input layer and first hidden layer
    # gradient[1] - gradient with respect to weights between first and second hidden layer
    # ...
    # gradient[-1] - gradient with respect to weights between last hidden layer and output layer
    gradients = []
    for i in range(0, num_layers-1):
        gradients.append(np.sum(np.transpose(np.dot(outputs[i].T, errors[-i-1]),(1,0,2)), axis=0)/N)

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