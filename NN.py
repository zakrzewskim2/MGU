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
    return sigmoid(x) * (1 - sigmoid(x)) 
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

#%%
def activation(x, derivative=False):
    return activation_function(x,ActivationFunction.SOFTMAX, derivative)

def out_activation(x, derivative=False):
    return activation_function(x, ActivationFunction.LINEAR, derivative)

def error_function(y, y_exp, derivative = False):
    if (derivative == True):
        return (2 * (y - y_exp) / y_exp.shape[1])
    else:
        return ((y - y_exp)**2).mean()
#%%
import numpy as np
import pandas as pd

train = pd.read_csv("data\classification\data.simple.train.100.csv")
test = pd.read_csv("data\classification\data.simple.test.100.csv")
X_train = train.iloc[:, :-1]
y_train = train["cls"]
X_test = np.array(test.iloc[:, :-1])
y_test = np.array([test["cls"]]).T

# inputs - each row is an input to MLP
s = 100
X = np.array(X_train)[0:s,:]
# outputs - each row is an expected output for the corresponding input
y = np.array([y_train]).T[0:s,:]

# N - number of input vectors
N = X.shape[0]

# number of layers (all of them!)
num_layers = 5

# number of nodes in each layer
layer_lengths = np.array([X.shape[1], 5, 5, 3, y.shape[1]])
if layer_lengths.shape[0] != num_layers:
    print("Error! Number of layers undefined!")

# %%
import matplotlib.pyplot as plt
import networkx as nx

visualize = True
G = nx.Graph()

np.random.seed(1)
weight_matrices = []
for i in range(0, num_layers-1):
    nrows = layer_lengths[i] + 1
    ncols = layer_lengths[i + 1]
    # Initial weights from [-1,1], with first row as bias
    weight_matrices.append(2*np.random.random((nrows, ncols)) - 1)

if visualize:
    # Init visualization nodes
    vis_mappings = []

    # bias node
    G.add_node(0, pos = (4, 0))
    node_num = 1

    for i in range(num_layers):
        new_nodes = range(node_num, node_num + layer_lengths[i])
        vis_mappings.append(np.array(new_nodes))
        for index, node in enumerate(new_nodes):
            G.add_node(node, pos=(4 * (i + 1), 2 * index + 2))
        node_num += len(new_nodes)

    def draw_graph(G, with_colorbar = False):
        # Init visualization edges
        for i in range(num_layers - 1):
            # bias edges
            for in_node in range(layer_lengths[i + 1]):
                G.add_edge(0, vis_mappings[i + 1][in_node],
                    weight = round(weight_matrices[i][0, in_node], 2))

            # next layer edges
            for out_node in range(1, layer_lengths[i] + 1):
                for in_node in range(layer_lengths[i + 1]):
                    G.add_edge(vis_mappings[i][out_node - 1], vis_mappings[i + 1][in_node],
                        weight = round(weight_matrices[i][out_node, in_node], 2))

        # draw graph
        pos = nx.get_node_attributes(G, 'pos')
        weights = nx.get_edge_attributes(G, 'weight')
        # nx.draw_networkx(G, pos)
        # nx.draw(G, pos)

        nodes = G.nodes()
        colors = []
        weights = nx.get_edge_attributes(G, 'weight')
        for node in G.nodes():
            if node > layer_lengths[0]:
                colors.append(weights[(0, node)])
            else:
                colors.append(0)

        nc = nx.draw_networkx_nodes(G, pos, nodelist=nodes, node_color=colors, 
            with_labels=True, node_size=100, cmap=plt.cm.viridis)
        
        if with_colorbar:
            plt.colorbar(nc)
        
        for edge in G.edges(data='weight'):
            nx.draw_networkx_edges(G, pos, edgelist=[edge], width=abs(edge[2]), edge_color = 'r' if edge[2] > 0 else 'b')

        # labels = nx.get_edge_attributes(G, 'weight')
        # nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)

        plt.pause(0.05)
    
    draw_graph(G, with_colorbar = True)

# number of iterations
num_iterations = 25000
# learning rate
alpha = 0.1

for j in range(num_iterations):
    # output of each layer
    # outputs[0] - inputs with ones in first column (input for bias)
    # outputs[1] - output of first hidden layer, with ones in first column
    # ...
    # outputs[-1] - output of last layer (without ones)
    outputs = []
    not_activated_outputs = []
    outputs.append(np.hstack((np.ones((X.shape[0], 1)), X)))
    not_activated_outputs.append(outputs[0])
    for i in range(0, num_layers-2):
        outputs.append(np.hstack((np.ones((X.shape[0], 1)), activation(np.dot(outputs[i], weight_matrices[i])))))
        not_activated_outputs.append(np.hstack((np.ones((X.shape[0], 1)), np.dot(outputs[i], weight_matrices[i]))))
    outputs.append(np.dot(outputs[-1], weight_matrices[-1]))
    not_activated_outputs.append(outputs[-1])

    # error term for each layer
    # errors[0] - error for last layer
    # errors[1] - error for last hidden layer
    # ...
    # errors[-1] - error for first hidden layer
    errors = []
    err = (np.transpose(error_function(out_activation(outputs[-1]), y, True)[None,:,:],(0,2,1)) * \
        np.transpose(out_activation(outputs[-1], True),(0,2,1))).sum(axis=1)
    errors.append(err.T)
    for i in range(0, num_layers-2):
        errors.append((np.dot(weight_matrices[-i-1][1:,:], errors[i].T)[None,:,:] * \
            np.transpose(activation(not_activated_outputs[-i-2][:, 1:], True),(0,2,1))).sum(axis=1).T)

    # gradient for each matrix of weights
    # gradient[0] - gradient with respect to weights between input layer and first hidden layer
    # gradient[1] - gradient with respect to weights between first and second hidden layer
    # ...   
    # gradient[-1] - gradient with respect to weights between last hidden layer and output layer
    gradients = []
    for i in range(0, num_layers-1):
        gradients.append(np.dot(outputs[i].T, errors[-i-1])/N)

    # Adjusting weights
    for i in range(0, num_layers-1):
        weight_matrices[i] += -alpha * gradients[i]

    if j % 1000 == 0:
        print("Error:", error_function(y, out_activation(outputs[-1])))

    if visualize and j % 1000 == 0:
        draw_graph(G)

if visualize:
    plt.show()

# %%
result = []
result.append(np.hstack((np.ones((X_test.shape[0], 1)), X_test)))
for i in range(0, num_layers-2):
    result.append(np.hstack((np.ones((X_test.shape[0], 1)), activation(np.dot(result[i], weight_matrices[i])))))
result.append(np.dot(result[-1], weight_matrices[-1]))

with np.printoptions(precision=3, suppress=True):
    print("Expected output:\n", y_test.T)
    print("Output After Training:\n", out_activation(result[-1]).T)
    print("Error:\n", error_function(y_test, out_activation(result[-1])))
# %%