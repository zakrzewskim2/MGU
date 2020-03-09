from .base_nn import BackpropagationNeuralNetwork
from .params import Config

import mlp.activation_functions as activation_functions
import mlp.error_functions as error_functions

import numpy as np
import pandas as pd

class MLPRegressor(BackpropagationNeuralNetwork):
    def __init__(self, activation_function = activation_functions.linear, \
            error_function = error_functions.mean_squared, \
            hidden_layers = [5, 5, 3], bias = True, batch_portion = 0.5, \
            num_iterations = 100000, eta = 0.1, moment = 0):
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

        super().__init__(config)

    def fit(self, X, y, save_result=False, random_seed=12369666):
        # X is 2D - np.ndarray / pd.DataFrame
        X = self.__check_if_X_is_valid(X)

        # y is 1D - np.ndarray / pd.Series
        y = self.__check_if_y_is_valid(y)

        y = y.reshape(-1, 1)        
        super().fit(X, y, save_result=save_result, random_seed=random_seed)
        return self

    def predict(self, X):
        X = self.__check_if_X_is_valid(X)
        
        return super().predict(X)

    def score(self, X, y):
        # returns standard error of the estimate
        X = self.__check_if_X_is_valid(X)
        y = self.__check_if_y_is_valid(y)

        predicted_y = self.predict(X)

        def standard_error_of_the_estimate(predicted_y, y):
            n = y.shape[0]
            return np.sqrt(np.sum((predicted_y - y)**2) / n)

        return standard_error_of_the_estimate(predicted_y, y)
    

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
            raise ValueError('X and y should be of type pandas.DataFrame or np.ndarray')
        else:
            if y.ndim != 1:
                raise ValueError('y should be 1D')
        
        return y