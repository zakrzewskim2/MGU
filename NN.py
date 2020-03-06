# %%
import joblib
import pandas as pd
import numpy as np
from params import Config

config = Config()

activation = config.activation_function
out_activation = config.activation_function
error_function = config.error_function

# %%
train = pd.read_csv("data\classification\data.simple.train.100.csv")
test = pd.read_csv("data\classification\data.simple.test.100.csv")
X_train, y_train = train.iloc[:, :-1], train["cls"]
X_test, y_test = test.iloc[:, :-1], test["cls"]

# inputs - each row is an input to MLP
X = X_train.values


def one_hot_encode(y, min_class, max_class):
    return np.identity(max_class - min_class + 1)[y - min_class, :]


def one_hot_decode(encoded_y, min_class):
    return np.argmax(encoded_y, axis=1) + min_class


# outputs - each row is an expected output for the corresponding input
y = y_train.values
y = one_hot_encode(y, min_class=np.min(y),
                   max_class=np.max(y))
y_test = one_hot_encode(y_test, min_class=np.min(y_test.values),
                        max_class=np.max(y_test.values))

# %%
# N - number of input vectors
N = X.shape[0]

# number of layers (all of them!)
num_layers = len(config.hidden_layers) + 2

# number of nodes in each layer
layer_lengths = np.array([X.shape[1]] + config.hidden_layers + [y.shape[1]])
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
num_iterations = config.num_iterations
# learning rate
alpha = config.eta


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
        activated_output = layer_output(outputs[i], weight_matrices[i],
                                        activate=True)
        not_activated_output = layer_output(outputs[i], weight_matrices[i],
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
    'config': config,
    'layer_lengths': layer_lengths
}
joblib.dump(training_data, 'training_data.joblib')

# %%
result = []
result.append(append_ones_column(X_test))
for i in range(0, num_layers-2):
    activated_result = layer_output(result[i], weight_matrices[i],
                                    activate=True)
    result.append(activated_result)
result.append(np.dot(result[-1], weight_matrices[-1]))

with np.printoptions(precision=3, suppress=True):
    print("Expected output:\n", y_test.T)
    print("Output After Training:\n", out_activation(result[-1]))
    print("Error:\n", error_function(out_activation(result[-1]), y_test))

# %%


class BackpropagationNeuralNetwork():
    def __init__(self, config: Config):
        self.config: Config = config
        self.num_iterations = self.config.num_iterations
        self.alpha = self.config.eta

        self.error_function = self.config.error_function
        self.activation = self.config.activation_function
        self.out_activation = self.config.activation_function

    def fit(self, X, y, random_seed=12369666, save_result=False):
        self.__initialize_structures(X, y)
        self.__initialize_weights(random_seed)

        if save_result:
            self.weight_history = [self.__current_weights_deep_copy()]

        for j in range(self.num_iterations):
            self.__calculate_outputs(X)
            self.__calculate_errors(X, y)
            self.__calculate_gradients()
            self.__adjust_weights()

            if j % 1000 == 0:
                print("Error:",
                      self.error_function(self.out_activation(self.outputs[-1]), y))

            if save_result:
                self.weight_history.append(self.__current_weights_deep_copy())

        if save_result:
            self.__serialize_training_info()

    def predict(self, X):
        result = []
        result.append(self.__append_ones_column(X))
        for i in range(0, self.num_layers - 2):
            activated_result = self.__layer_output(result[i], self.weight_matrices[i],
                                                   activate=True)
            result.append(activated_result)
        result.append(np.dot(result[-1], self.weight_matrices[-1]))

        return self.out_activation(result[-1])

    def score(self, X, y, one_hot_decode_y=False):
        y_predicted = self.predict(X)
        if one_hot_decode_y:
            y_predicted = one_hot_decode(y_predicted, min_class = 1)
        
        print(np.mean(y == y_predicted))
        with np.printoptions(precision=3, suppress=True):
            print("Expected output:\n", y)
            print("Output After Training:\n", y_predicted)
            print("Error:\n", self.error_function(y_predicted, y))

    def __serialize_training_info(self):
        training_info = {
            'weight_history': self.weight_history,
            'config': self.config,
            'layer_lengths': self.layer_lengths
        }
        joblib.dump(training_data, 'training_info.joblib')

    def __initialize_structures(self, X, y):
        # N - number of input vectors
        self.N = X.shape[0]

        # number of nodes in each layer
        self.layer_lengths = np.array([X.shape[1]] +
                                      self.config.hidden_layers +
                                      [y.shape[1]])
        self.num_layers = len(self.layer_lengths)

    def __initialize_weights(self, random_seed):
        np.random.seed(random_seed)
        self.weight_matrices = []
        for i in range(0, self.num_layers - 1):
            nrows = self.layer_lengths[i] + 1
            ncols = self.layer_lengths[i + 1]

            # Initial weights from [-1,1], with first row as bias
            self.weight_matrices.append(
                2 * np.random.random((nrows, ncols)) - 1)

    def __append_ones_column(self, X):
        # X = [[2, 3], [4, 5]] -> X = [[1, 2, 3], [1, 4, 5]]
        return np.insert(X, 0, 1, axis=1)

    def __layer_i_train_output(self, i, activate=True):
        return self.__layer_output(self.outputs[i], self.weight_matrices[i])

    def __layer_output(self, inputs, weights, activate=True):
        base_output = np.dot(inputs, weights)
        if activate:
            return self.__append_ones_column(self.activation(base_output))
        else:
            return self.__append_ones_column(base_output)

    def __current_weights_deep_copy(self):
        return np.array([np.copy(x) for x in self.weight_matrices])

    def __calculate_outputs(self, X):
        # output of each layer
        # outputs[0] - inputs with ones in first column (input for bias)
        # outputs[1] - output of first hidden layer, with ones in first column
        # ...
        # outputs[-1] - output of last layer (without ones)
        self.outputs = []
        self.not_activated_outputs = []

        self.outputs.append(self.__append_ones_column(X))
        self.not_activated_outputs.append(self.outputs[0])

        for i in range(self.num_layers - 2):
            activated_output = self.__layer_i_train_output(i)
            not_activated_output = self.__layer_i_train_output(
                i, activate=False)
            self.outputs.append(activated_output)
            self.not_activated_outputs.append(not_activated_output)

        self.outputs.append(
            np.dot(self.outputs[-1], self.weight_matrices[-1])
        )
        self.not_activated_outputs.append(self.outputs[-1])

    def __calculate_errors(self, X, y):
        # error term for each layer
        # errors[0] - error for last layer
        # errors[1] - error for last hidden layer
        # ...
        # errors[-1] - error for first hidden layer

        self.errors = []
        err = (
            np.transpose(self.error_function(self.out_activation(self.outputs[-1]), y, True)[None, :, :], (0, 2, 1)) *
            np.transpose(self.out_activation(
                self.outputs[-1], True), (0, 2, 1))
        ).sum(axis=1)
        self.errors.append(err.T)

        for i in range(0, self.num_layers - 2):
            self.errors.append(
                (np.dot(self.weight_matrices[-i-1][1:, :], self.errors[i].T)[None, :, :] *
                 np.transpose(self.activation(
                     self.not_activated_outputs[-i-2][:, 1:], True), (0, 2, 1))
                 ).sum(axis=1).T
            )

    def __calculate_gradients(self):
        # gradient for each matrix of weights
        # gradient[0] - gradient with respect to weights between input layer and first hidden layer
        # gradient[1] - gradient with respect to weights between first and second hidden layer
        # ...
        # gradient[-1] - gradient with respect to weights between last hidden layer and output layer
        self.gradients = []
        for i in range(self.num_layers - 1):
            gradient = np.dot(self.outputs[i].T, self.errors[-i-1]) \
                / self.N
            self.gradients.append(gradient)

    def __adjust_weights(self):
        # Adjusting weights
        for i in range(self.num_layers - 1):
            self.weight_matrices[i] += -self.alpha * self.gradients[i]


# %%
config = Config()
config.num_iterations = 10000
nn = BackpropagationNeuralNetwork(config)
nn.fit(X, y)

# %%
nn.score(X_test.values, y_test)

# %%
nn.score(X_test.values, test["cls"], one_hot_decode_y = True)

# %%
