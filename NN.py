# %%
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import numpy as np


def activation(x, derivative=False):
    if (derivative == True):
        return x * (1 - x)
    else:
        return 1 / (1 + np.exp(-x))


def error_function(y, y_exp, derivative=False):
    if (derivative == True):
        return 2 * (y - y_exp) / y_exp.shape[1]
    else:
        return ((y - y_exp)**2).mean()


def out_activation(x, derivative=False):
    if (derivative == True):
        return 1
    else:
        return x


# %%
train = pd.read_csv("data\classification\data.simple.train.100.csv")
test = pd.read_csv("data\classification\data.simple.test.100.csv")
X_train = train.iloc[:, :-1]
y_train = train["cls"]
X_test = np.array(test.iloc[:, :-1])
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

# %%
visualize = True
G = nx.Graph()

np.random.seed(1)
weigth_matrices = []
for i in range(0, num_layers-1):
    nrows = layer_lengths[i] + 1
    ncols = layer_lengths[i + 1]
    # Initial weights from [-1,1], with first row as bias
    weigth_matrices.append(2*np.random.random((nrows, ncols)) - 1)

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

    # Init visualization edges
    for i in range(num_layers - 1):
        # bias edges
        for in_node in range(layer_lengths[i + 1]):
            G.add_edge(0, vis_mappings[i + 1][in_node],
                weight = round(weigth_matrices[i][0, in_node], 2))

        # next layer edges
        for out_node in range(1, layer_lengths[i] + 1):
            for in_node in range(layer_lengths[i + 1]):
                G.add_edge(vis_mappings[i][out_node - 1], vis_mappings[i + 1][in_node],
                    weight = round(weigth_matrices[i][out_node, in_node], 2))

    # draw graph
    pos = nx.get_node_attributes(G, 'pos')
    nx.draw_networkx(G, pos)
    # nx.draw(G, pos)

    for edge in G.edges(data='weight'):
        nx.draw_networkx_edges(G, pos, edgelist=[edge], width=5*edge[2])

    labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)

    plt.pause(0.05)

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
        outputs.append(np.hstack((np.ones((X.shape[0], 1)), activation(
            np.dot(outputs[i], weigth_matrices[i])))))
    outputs.append(np.dot(outputs[-1], weigth_matrices[-1]))

    # error term for each layer
    # errors[0] - error for last layer
    # errors[1] - error for last hidden layer
    # ...
    # errors[-1] - error for first hidden layer
    errors = []
    errors.append(out_activation(outputs[-1], True) *
                  error_function(out_activation(outputs[-1]), y, True))
    for i in range(0, num_layers-2):
        errors.append(activation(outputs[-i-2][:, 1:], True) *
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

    if j % 1000 == 0:
        print("Error:", error_function(y, outputs[-1]))

    if visualize and j % 100 == 0:
        # Init visualization edges
        for i in range(num_layers - 1):
            # bias edges
            for in_node in range(layer_lengths[i + 1]):
                G.add_edge(0, vis_mappings[i + 1][in_node],
                    weight = round(weigth_matrices[i][0, in_node], 2))

            # next layer edges
            for out_node in range(1, layer_lengths[i] + 1):
                for in_node in range(layer_lengths[i + 1]):
                    G.add_edge(vis_mappings[i][out_node - 1], vis_mappings[i + 1][in_node],
                        weight = round(weigth_matrices[i][out_node, in_node], 2))

        # draw graph
        pos = nx.get_node_attributes(G, 'pos')
        nx.draw_networkx(G, pos)

        for edge in G.edges(data='weight'):
            nx.draw_networkx_edges(G, pos, edgelist=[edge], width=5*edge[2])

        labels = nx.get_edge_attributes(G, 'weight')
        nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)

        plt.pause(0.05)

if visualize:
    plt.show()

# %%
result = []
result.append(np.hstack((np.ones((X_test.shape[0], 1)), X_test)))
for i in range(0, num_layers-2):
    result.append(np.hstack((np.ones((X_test.shape[0], 1)), activation(
        np.dot(result[i], weigth_matrices[i])))))
result.append(np.dot(result[-1], weigth_matrices[-1]))

with np.printoptions(precision=3, suppress=True):
    print("Expected output:\n", y_test.T)
    print("Output After Training:\n", result[-1].T)
    print("Error:\n", error_function(y_test, result[-1]))