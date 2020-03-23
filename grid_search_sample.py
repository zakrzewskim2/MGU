# %%
import pandas as pd

from mlp import MLPRegressor, activation_functions, \
    error_functions, GridSearch

param_grid = {
    'hidden_layers': [[], [5, 5, 5, 5]],
    'num_iterations': [10, 100, 1000],
    'eta': [0.005, 0.1],
    'batch_portion': [0.1, 0.5, 1],
    'bias': [True, False],
    'activation_function': [activation_functions.sigmoid, \
        activation_functions.tanh],
    'error_function': [error_functions.mean_squared, \
        error_functions.mean],
    'moment': [0, 0.2, 0.5]
}

gs = GridSearch(MLPRegressor(random_seed = 12369666), param_grid)

train = pd.read_csv("data/regression/data.cube.train.100.csv")
test = pd.read_csv("data/regression/data.cube.test.100.csv")
X_train, y_train = train.iloc[:, :-1], train.y
X_test, y_test = test.iloc[:, :-1], test.y
gs.fit(X_train, y_train, X_test, y_test)

#%%
f = open("grid_search_scores.txt","w+")
f.write(str(gs.param_scores_))
f.close()

# %%
print(gs.param_scores_)

# %%
print(gs.best_params_)
