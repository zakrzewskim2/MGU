# %%
import pandas as pd

from grid_search import GridSearch
from mlp import MLPRegressor
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
    'error_function': [error_functions.mean, \
        error_functions.mean_squared],
    'moment': [0, 0.05, 0.1, 0.2, 0.5]
}

gs = GridSearch(MLPRegressor(), param_grid)

train = pd.read_csv("data/regression/data.cube.train.100.csv")
test = pd.read_csv("data/regression/data.cube.test.100.csv")
X_train, y_train = train.iloc[:, :-1], train.y
X_test, y_test = test.iloc[:, :-1], test.y
gs.fit(X_train, y_train, X_test, y_test)

#%%
f= open("scores.txt","w+")
f.write(str(gs.param_scores_))
f.close()
#print(gs.param_scores_)

# %%
gs.best_params_

# %%
max(gs.param_scores_, \
            key = lambda p: p['score'])

# %%
