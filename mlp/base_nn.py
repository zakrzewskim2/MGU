from .params import Config

import numpy as np
import pandas as pd
import joblib
import random

class BackpropagationNeuralNetwork():
    def __init__(self, config: Config):
        self.config: Config = config
        
        self.hidden_layers = config.hidden_layers
        self.bias = config.bias
        self.batch_portion = config.batch_portion
        self.moment = config.moment
        self.num_iterations = config.num_iterations
        self.eta = config.eta

        self.error_function = config.error_function
        self.activation = config.activation_function
        self.out_activation = config.out_activation_function

    def fit(self, X, y, random_seed=12369666, serialize_path=None):
        if type(X) == pd.Series:
            X = X.values
        if type(y) == pd.Series:
            y = y.values
        
        random.seed(random_seed)
        self.__initialize_structures(X, y)
        self.__initialize_weights(random_seed)

        self.previous_weights_diff = self.__deep_zeros_like_copy()
        self.weight_history = [self.__current_weights_deep_copy()]

        batch_size = int(self.batch_portion * self.N)

        for j in range(self.num_iterations):
            indices = random.sample(range(0, self.N), batch_size)
            batch_X = X[indices]
            batch_y = y[indices]
            self.__calculate_outputs(batch_X)
            self.__calculate_errors(batch_X, batch_y)
            self.__calculate_gradients()
            self.__adjust_weights()

            if j % 1000 == 0:
                self.eta *= 0.95
                # with np.printoptions(precision=3, suppress=True):
                #     print(f"Iter: {j}/{self.num_iterations} Error:",
                #       self.error_function(self.out_activation(self.outputs[-1]), batch_y))

            self.weight_history.append(self.__current_weights_deep_copy())

        if serialize_path is not None:
            self.__serialize_training_info(serialize_path)

    def predict(self, X):
        if type(X) == pd.Series:
            X = X.values
        
        result = []
        result.append(self.__append_bias_input_column(X))
        for i in range(0, self.num_layers - 2):
            activated_result = self.__layer_output(result[i], self.weight_matrices[i],
                                                   activate=True)
            result.append(activated_result)
        result.append(np.dot(result[-1], self.weight_matrices[-1]))

        return self.out_activation(result[-1])

    def score(self, X, y):
        if type(X) == pd.Series:
            X = X.values
        if type(y) == pd.Series:
            y = y.values
        
        y_predicted = self.predict(X)
        
        print(np.mean(y == y_predicted))
        with np.printoptions(precision=3, suppress=True):
            print("Expected output:\n", y)
            print("Output After Training:\n", y_predicted)
            print("Error:\n", self.error_function(y_predicted, y))

    def error_by_iteration(self, X, y):
        if type(X) == pd.Series:
            X = X.values
        if type(y) == pd.Series:
            y = y.values

        iteration_errors = []
        for weights in self.weight_history:
            outputs, _ = \
                self.__calculate_outputs_with_weights(X, weights)
            iteration_errors.append(\
                self.error_function(self.out_activation(outputs[-1]), y)\
            )
        return iteration_errors

    def __serialize_training_info(self, serialize_path):
        print('Serializing result...')
        training_data = {
            'weight_history': self.weight_history,
            'config': self.config,
            'layer_lengths': self.layer_lengths
        }
        joblib.dump(training_data, serialize_path)

    def __initialize_structures(self, X, y):
        # N - number of input vectors
        self.N = X.shape[0]

        # number of nodes in each layer
        self.layer_lengths = np.array([X.shape[1]] +
                                      self.hidden_layers +
                                      [y.shape[1]])
        self.num_layers = len(self.layer_lengths)

    def __initialize_weights(self, random_seed):
        np.random.seed(random_seed)
        self.weight_matrices = []
        for i in range(0, self.num_layers - 1):
            nrows = self.layer_lengths[i]
            ncols = self.layer_lengths[i + 1]

            # Initial weights from [-1,1], with first row as bias
            weights = 2 * np.random.random((nrows, ncols)) - 1
            bias = 2 * np.random.random((1, ncols)) - 1 if self.bias \
                    else np.zeros((1, ncols))
            weights = np.insert(weights, 0, bias, axis=0)
            self.weight_matrices.append(weights)

    def __append_bias_input_column(self, X):
        # X = [[2, 3], [4, 5]] -> X = [[1, 2, 3], [1, 4, 5]]
        if self.bias:
            return np.insert(X, 0, 1, axis=1)
        else:
            return np.insert(X, 0, 0, axis=1)

    def __layer_i_train_output(self, i, activate=True):
        return self.__layer_output(self.outputs[i], self.weight_matrices[i])

    def __layer_output(self, inputs, weights, activate=True):
        base_output = np.dot(inputs, weights)
        if activate:
            return self.__append_bias_input_column(self.activation(base_output))
        else:
            return self.__append_bias_input_column(base_output)

    def __current_weights_deep_copy(self):
        return [np.copy(x) for x in self.weight_matrices]

    def __deep_zeros_like_copy(self):
        return [np.array(np.zeros_like(x)) for x in self.weight_matrices]

    def __calculate_outputs(self, X):
        self.outputs, self.not_activated_outputs = \
            self.__calculate_outputs_with_weights(X, self.weight_matrices)

    def __calculate_outputs_with_weights(self, X, weights):
        # output of each layer
        # outputs[0] - inputs with ones in first column (input for bias)
        # outputs[1] - output of first hidden layer, with ones in first column
        # ...
        # outputs[-1] - output of last layer (without ones)
        outputs = []
        not_activated_outputs = []

        outputs.append(self.__append_bias_input_column(X))
        not_activated_outputs.append(outputs[0])

        for i in range(self.num_layers - 2):
            activated_output = self.__layer_output(outputs[i], weights[i], \
                activate=True)
            not_activated_output = self.__layer_output(outputs[i], weights[i], \
                activate=False)
            outputs.append(activated_output)
            not_activated_outputs.append(not_activated_output)

        outputs.append(
            np.dot(outputs[-1], weights[-1])
        )
        not_activated_outputs.append(outputs[-1])
        
        return outputs, not_activated_outputs

    def __calculate_errors(self, X, y):
        # error term for each layer
        # errors[0] - error for last layer
        # errors[1] - error for last hidden layer
        # ...
        # errors[-1] - error for first hidden layer

        self.errors = self.__calculate_errors_with_outputs(X, y, self.weight_matrices,\
            self.outputs, self.not_activated_outputs)

    def __calculate_errors_with_outputs(self, X, y, weights, outputs, not_activated_outputs):
        errors = []
        err = (
            np.transpose(self.error_function(self.out_activation(outputs[-1]), y, True)[None, :, :], (0, 2, 1)) *
            np.transpose(self.out_activation(
                outputs[-1], True), (0, 2, 1))
        ).sum(axis=1)
        errors.append(err.T)

        for i in range(0, self.num_layers - 2):
            errors.append(
                (np.dot(weights[-i-1][1:, :], errors[i].T)[None, :, :] *
                 np.transpose(self.activation(
                     not_activated_outputs[-i-2][:, 1:], True), (0, 2, 1))
                 ).sum(axis=1).T
            )

        return errors

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
            delta_w = self.eta * self.gradients[i] + \
                self.moment * self.previous_weights_diff[i]
            self.weight_matrices[i] += -delta_w
            self.previous_weights_diff[i] = delta_w