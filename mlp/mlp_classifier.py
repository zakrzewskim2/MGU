from .base_nn import BackpropagationNeuralNetwork
from .params import Config

import mlp.activation_functions as activation_functions
import mlp.error_functions as error_functions

import numpy as np
import pandas as pd

class MLPClassifier(BackpropagationNeuralNetwork):
    def __init__(self, activation_function = activation_functions.tanh, \
            error_function = error_functions.cross_entropy, \
            hidden_layers = [20, 20], bias = True, batch_portion = 0.7, \
            num_iterations = 40000, eta = 0.1, moment = 0, \
            random_seed=None):
        config = Config()
        config.out_activation_function = activation_functions.softmax

        config.activation_function = activation_function
        config.error_function = error_function
        config.hidden_layers = hidden_layers
        config.bias = bias
        config.batch_portion = batch_portion
        config.num_iterations = num_iterations
        config.eta = eta
        config.moment = moment

        super().__init__(random_seed, config)

    def fit(self, X, y, serialize_path=None):
        # X is 2D - np.ndarray / pd.DataFrame
        X = self.__check_if_X_is_valid(X)

        # y is 1D - np.ndarray / pd.Series
        y = self.__check_if_y_is_valid(y)
        
        self.min_class = np.min(y)
        self.max_class = np.max(y)
        encoded_y = self.__one_hot_encode(y, min_class=self.min_class, \
            max_class=self.max_class)
        
        super().fit(X, encoded_y, serialize_path=serialize_path)
        return self

    def predict(self, X):
        X = self.__check_if_X_is_valid(X)
        
        encoded_y = super().predict(X)
        return self.__one_hot_decode(encoded_y, self.min_class)

    def score(self, X, y):
        X = self.__check_if_X_is_valid(X)
        y = self.__check_if_y_is_valid(y)

        predicted_y = self.predict(X)
        return np.mean(y == predicted_y)

    def confusion_matrix(self, X, y):
        X = self.__check_if_X_is_valid(X)
        y = self.__check_if_y_is_valid(y)
        predicted_y = self.predict(X)

        size = self.max_class - self.min_class + 1
        confusion_matrix = np.zeros((size, size), dtype=int)
        np.add.at(confusion_matrix, \
            [y - self.min_class, predicted_y - self.min_class], \
            1)
        return confusion_matrix

    def error_by_iteration(self, X, y):
        X = self.__check_if_X_is_valid(X)
        y = self.__check_if_y_is_valid(y)
        
        encoded_y = self.__one_hot_encode(y, self.min_class, self.max_class)
        return super().error_by_iteration(X, encoded_y)

    def __one_hot_encode(self, y, min_class, max_class):
        return np.identity(max_class - min_class + 1)[y - min_class, :]

    def __one_hot_decode(self, encoded_y, min_class):
        return np.argmax(encoded_y, axis=1) + min_class

    def __check_if_X_is_valid(self, X):
        if type(X) == pd.DataFrame:
            X = X.values

        if type(X) != np.ndarray:
            raise ValueError('X should be of type pandas.DataFrame or np.ndarray')
        else:
            if X.ndim != 2:
                raise ValueError('X should be 2D')
        
        return X

    def __check_if_y_is_valid(self, y):
        if type(y) == pd.Series:
            y = y.values

        if type(y) != np.ndarray:
            raise ValueError('y should be of type pandas.DataFrame or np.ndarray')
        else:
            if y.ndim != 1:
                raise ValueError('y should be 1D')
        
        return y