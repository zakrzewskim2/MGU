# %%
import pandas as pd

from grid_search import GridSearch
from mlp import MLPClassifier
from mlp import activation_functions
from mlp import error_functions

param_grid = {
    'hidden_layers': [[], [5], [5, 5], [5, 5, 5], [5, 5, 5, 5]],
    'num_iterations': [10, 100, 1000],
    'eta': [0.0005, 0.005, 0.05, 0.5],
    'batch_portion': [0.1, 0.3, 0.5, 0.7, 0.9, 1],
    'bias': [True, False],
    'activation_function': [activation_functions.softmax, \
        activation_functions.sigmoid, \
        activation_functions.tanh, \
        activation_functions.linear],
    'error_function': [error_functions.cross_entropy, \
        error_functions.max_error, \
        error_functions.mean, \
        error_functions.mean_squared],
    'moment': [0, 0.05, 0.1, 0.2, 0.5]
}

gs = GridSearch(MLPClassifier(), param_grid)

train = pd.read_csv("data/projekt1_test/Classification/data.circles.train.100.csv")
test = pd.read_csv("data/projekt1_test/Classification/data.circles.test.100.csv")
X_train, y_train = train.iloc[:, :-1], train.cls
X_test, y_test = test.iloc[:, :-1], test.cls

gs.fit(X_train, y_train)

# %%
gs.param_scores_

# %%
gs.best_params_

# %%
