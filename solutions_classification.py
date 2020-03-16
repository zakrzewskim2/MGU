# %%
import joblib
import pandas as pd

from mlp.mlp_classifier import MLPClassifier
import mlp.activation_functions as activation_functions
import mlp.error_functions as error_functions

from visualizer import Visualizer
vis = Visualizer()

def process_classification_dataset(name, clf, size = 100, datasets_path_format = "data/projekt1_test/Classification/data.{}.{}.{}.csv"):
    train = pd.read_csv(datasets_path_format.format(name, 'train', size))
    test = pd.read_csv(datasets_path_format.format(name, 'test', size))

    X_train, y_train = train.iloc[:, :-1], train.cls
    X_test, y_test = test.iloc[:, :-1], test.cls

    clf.fit(X_train, y_train)
    print(clf.score(X_test, y_test))

    joblib.dump(clf, f'results/{name}-{size}-clf.joblib')
    vis.plot_classification_result(clf, \
        X_test.x, X_test.y, y_test, \
        save_path = f'results/{name}-{size}-result.png')
    vis.plot_train_test_error(clf, \
        X_train, y_train, X_test, y_test, \
        save_path = f'results/{name}-{size}-error.png')

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
    hidden_layers = [5, 5, 3], bias = True, batch_portion = 0.5, \
    num_iterations = 40000, eta = 0.1, moment = 0)

process_classification_dataset('simple', clf, \
    datasets_path_format='data/classification/data.{}.{}.{}.csv')

# %% three_gauss
clf = MLPClassifier(activation_function = \
        activation_functions.sigmoid, \
    error_function = error_functions.cross_entropy, \
    hidden_layers = [5, 5, 3], bias = True, batch_portion = 0.5, \
    num_iterations = 40000, eta = 0.1, moment = 0)

process_classification_dataset('three_gauss', clf, \
    datasets_path_format='data/classification/data.{}.{}.{}.csv')

# %%
