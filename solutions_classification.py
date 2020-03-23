# %%
import joblib
import pandas as pd
import numpy as np

from mlp import MLPClassifier, activation_functions, \
    error_functions, Visualizer

vis = Visualizer()

def process_classification_dataset(name, clf, draw = True, \
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
    score = clf.score(X_test, y_test)
    print('Accuracy:', score)

    if draw:
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

    return score

# %% circles
scores = []
for i in range(10):
    clf = MLPClassifier(activation_function = \
            activation_functions.tanh, \
        error_function = error_functions.cross_entropy, \
        hidden_layers = [10, 10, 10, 10], bias = True, batch_portion = 0.84, \
        num_iterations = 5000, eta = 0.4155, moment = 0.5155, \
        random_seed = 12369666 + i)

    score = process_classification_dataset('circles', clf, \
        draw = True if i == 0 else False)
    
    scores.append(100 * score)
print(f'circles: mean - {round(np.mean(scores), 2)}, ' + \
    f'std - {round(np.std(scores), 2)}')

# %% XOR
scores = []
for i in range(10):
    clf = MLPClassifier(activation_function = \
            activation_functions.tanh, \
        error_function = error_functions.cross_entropy, \
        hidden_layers = [10, 10], bias = True, batch_portion = 0.7, \
        num_iterations = 10000, eta = 0.1, moment = 0, \
        random_seed = 12369666 + i)

    score = process_classification_dataset('XOR', clf, \
        draw = True if i == 0 else False)
    
    scores.append(100 * score)
print(f'XOR: mean - {round(np.mean(scores), 2)}, ' + \
    f'std - {round(np.std(scores), 2)}')

# %% noisyXOR
scores = []
for i in range(10):
    clf = MLPClassifier(activation_function = \
            activation_functions.tanh, \
        error_function = error_functions.cross_entropy, \
        hidden_layers = [5,5,3], bias = True, batch_portion = 0.7, \
        num_iterations = 10000, eta = 0.1, moment = 0, \
        random_seed = 12369666 + i)

    score = process_classification_dataset('noisyXOR', clf, \
            draw = True if i == 0 else False)
    
    scores.append(100 * score)
print(f'noisyXOR: mean - {round(np.mean(scores), 2)}, ' + \
    f'std - {round(np.std(scores), 2)}')

# %% simple
scores = []
for i in range(10):
    clf = MLPClassifier(activation_function = \
            activation_functions.sigmoid, \
        error_function = error_functions.cross_entropy, \
        hidden_layers = [], bias = True, batch_portion = 0.5, \
        num_iterations = 40000, eta = 0.1, moment = 0, \
        random_seed = 12369666 + i)

    score = process_classification_dataset('simple', clf, size=100, \
        datasets_path_format='data/classification/data.{}.{}.{}.csv', \
        draw = True if i == 0 else False)
    
    scores.append(100 * score)
print(f'simple: mean - {round(np.mean(scores), 2)}, ' + \
    f'std - {round(np.std(scores), 2)}')

# %% three_gauss
scores = []
for i in range(10):
    clf = MLPClassifier(activation_function = \
            activation_functions.sigmoid, \
        error_function = error_functions.cross_entropy, \
        hidden_layers = [5, 5, 3], bias = True, batch_portion = 0.5, \
        num_iterations = 40000, eta = 0.1, moment = 0, \
        random_seed = 12369666 + i)

    score = process_classification_dataset('three_gauss', clf, normalize=True, \
        datasets_path_format='data/classification/data.{}.{}.{}.csv', \
        draw = True if i == 0 else False)
    
    scores.append(100 * score)
print(f'three_gauss: mean - {round(np.mean(scores), 2)}, ' + \
    f'std - {round(np.std(scores), 2)}')

# %% windows
scores = []
for i in range(10):
    clf = MLPClassifier(activation_function = \
            activation_functions.tanh, \
        error_function = error_functions.cross_entropy, \
        hidden_layers = [30, 30], bias = True, batch_portion = 0.7, \
        num_iterations = 30000, eta = 0.2, moment = 0.2, \
        random_seed = 12369666 + i)

    score = process_classification_dataset('windows', clf, normalize=True, \
        datasets_path_format='data/classification/data.{}.{}.{}.csv', \
        plot_margin=0.25, plot_size=100, \
        draw = True if i == 0 else False)
    
    scores.append(100 * score)
print(f'windows: mean - {round(np.mean(scores), 2)}, ' + \
    f'std - {round(np.std(scores), 2)}')

# %%
