# %%
import joblib
import pandas as pd
import numpy as np

from mlp.mlp_classifier import MLPClassifier
import mlp.activation_functions as activation_functions
import mlp.error_functions as error_functions

from visualizer import Visualizer
vis = Visualizer()

def process_classification_dataset(name, clf, \
        normalize=False, size = 100, plot_margin = 1, plot_size = 40, \
        datasets_path_format = "data/projekt1_test/Classification/data.{}.{}.{}.csv"):
    train = pd.read_csv(datasets_path_format.format(name, 'train', size))
    test = pd.read_csv(datasets_path_format.format(name, 'test', size))

    X_train, y_train = train.iloc[:, :-1], train.cls
    X_test, y_test = test.iloc[:, :-1], test.cls

    if normalize:
        X_train /= np.max(np.abs(X_train))/4
        X_test /= np.max(np.abs(X_test))/4

    clf.fit(X_train, y_train)
    print(clf.score(X_test, y_test))

    joblib.dump(clf, \
        f'results/classification/{name}-{size}-clf.joblib')
    vis.plot_classification_dataset(X_train.x, X_train.y, \
        y_train, show = True, \
        save_path = f'results/classification/{name}-{size}-train.png')
    vis.plot_classification_result(clf, \
        X_test.x, X_test.y, y_test, show = True, \
        save_path = f'results/classification/{name}-{size}-result.png', \
        margin = plot_margin, grid_size = plot_size)
    vis.plot_train_test_error(clf, \
        X_train, y_train, X_test, y_test, \
        save_path = f'results/classification/{name}-{size}-error.png')

# %% circles
clf = MLPClassifier(activation_function = \
        activation_functions.tanh, \
    error_function = error_functions.cross_entropy, \
    hidden_layers = [20, 20], bias = True, batch_portion = 0.7, \
    num_iterations = 40000, eta = 0.1, moment = 0.2)

process_classification_dataset('circles', clf)

# %% XOR
clf = MLPClassifier(activation_function = \
        activation_functions.tanh, \
    error_function = error_functions.cross_entropy, \
    hidden_layers = [10, 10], bias = True, batch_portion = 0.7, \
    num_iterations = 10000, eta = 0.1, moment = 0)

process_classification_dataset('XOR', clf)

# %% noisyXOR
clf = MLPClassifier(activation_function = \
        activation_functions.tanh, \
    error_function = error_functions.cross_entropy, \
    hidden_layers = [5,5,3], bias = True, batch_portion = 0.7, \
    num_iterations = 10000, eta = 0.1, moment = 0)

process_classification_dataset('noisyXOR', clf)

# %% simple
clf = MLPClassifier(activation_function = \
        activation_functions.sigmoid, \
    error_function = error_functions.cross_entropy, \
    hidden_layers = [], bias = True, batch_portion = 0.5, \
    num_iterations = 40000, eta = 0.1, moment = 0)

process_classification_dataset('simple', clf, size=100, \
    datasets_path_format='data/classification/data.{}.{}.{}.csv')

# %% three_gauss
clf = MLPClassifier(activation_function = \
        activation_functions.sigmoid, \
    error_function = error_functions.cross_entropy, \
    hidden_layers = [5, 5, 3], bias = True, batch_portion = 0.5, \
    num_iterations = 40000, eta = 0.1, moment = 0)

process_classification_dataset('three_gauss', clf, normalize=True, \
    datasets_path_format='data/classification/data.{}.{}.{}.csv')

# %% windows
clf = MLPClassifier(activation_function = \
        activation_functions.tanh, \
    error_function = error_functions.cross_entropy, \
    hidden_layers = [30, 30], bias = True, batch_portion = 0.7, \
    num_iterations = 30000, eta = 0.2, moment = 0.2)

process_classification_dataset('windows', clf, normalize=True, \
    datasets_path_format='data/classification/data.{}.{}.{}.csv', \
    plot_margin=0.25, plot_size=100)

# %%
