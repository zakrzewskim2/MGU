# %%
import joblib
import pandas as pd
import numpy as np

from mlp.mlp_regressor import MLPRegressor
import mlp.activation_functions as activation_functions
import mlp.error_functions as error_functions

from visualizer import Visualizer
vis = Visualizer()

def process_regression_dataset(name, estimator, normalize = False, size = 100, datasets_path_format = "data/projekt1_test/Regression/data.{}.{}.{}.csv"):
    train = pd.read_csv(datasets_path_format.format(name, 'train', size))
    test = pd.read_csv(datasets_path_format.format(name, 'test', size))

    if normalize:
        train /= np.max(np.abs(train))
        test /= np.max(np.abs(test))

    X_train, y_train = train.iloc[:, :-1], train.y
    X_test, y_test = test.iloc[:, :-1], test.y

    estimator.fit(X_train, y_train)
    print(estimator.score(X_test, y_test))

    joblib.dump(estimator, \
        f'results/regression/{name}-{size}-clf.joblib')
    vis.plot_regression_result(estimator, \
        X_test.x, y_test, \
        show = True, \
        save_path = f'results/regression/{name}-{size}-result-test.png')
    vis.plot_regression_result(estimator, \
        X_train.x, y_train, \
        show = True, \
        save_path = f'results/regression/{name}-{size}-result-train.png')
    vis.plot_train_test_error(estimator, \
        X_train, y_train, X_test, y_test, \
        show = True, \
        save_path = f'results/regression/{name}-{size}-error.png')

# %% activation
estimator = MLPRegressor(activation_function = \
        activation_functions.sigmoid, \
    error_function = error_functions.mean_squared, \
    hidden_layers = [10], bias = True, batch_portion = 0.5, \
    num_iterations = 5000, eta = 0.1, moment = 0)

process_regression_dataset('activation', estimator, \
    datasets_path_format='data/regression/data.{}.{}.{}.csv')

# %% cube
estimator = MLPRegressor(activation_function = \
        activation_functions.sigmoid, \
    error_function = error_functions.mean_squared, \
    hidden_layers = [10, 10, 10], bias = True, batch_portion = 0.5, \
    num_iterations = 20000, eta = 0.01, moment = 0)

process_regression_dataset('cube', estimator, \
    datasets_path_format='data/regression/data.{}.{}.{}.csv')

# %% linear
estimator = MLPRegressor(activation_function = \
        activation_functions.tanh, \
    error_function = error_functions.mean_squared, \
    hidden_layers = [], bias = True, batch_portion = 0.8, \
    num_iterations = 1000, eta = 0.4, moment = 0)

process_regression_dataset('linear', estimator, size=100, \
    datasets_path_format='data/projekt1_test/Regression/data.{}.{}.{}.csv', \
    normalize=True)

# %% square
estimator = MLPRegressor(activation_function = \
        activation_functions.tanh, \
    error_function = error_functions.mean_squared, \
    hidden_layers = [8, 8, 8], bias = True, batch_portion = 0.8, \
    num_iterations = 10000, eta = 0.4, moment = 0)

process_regression_dataset('square', estimator, size=100, \
    datasets_path_format='data/projekt1_test/Regression/data.{}.{}.{}.csv', \
    normalize=True)

# %% multimodal
estimator = MLPRegressor(activation_function = \
        activation_functions.tanh, \
    error_function = error_functions.mean_squared, \
    hidden_layers = [8, 8, 8], bias = True, batch_portion = 0.8, \
    num_iterations = 10000, eta = 0.4, moment = 0)

process_regression_dataset('multimodal', estimator, size=100, \
    datasets_path_format='data/projekt1_test/Regression/data.{}.{}.{}.csv', \
    normalize=True)

# %%
