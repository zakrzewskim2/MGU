# %%
from copy import deepcopy

param_grid = {
    'hidden_layers': [[], [5], [5, 5], [5, 5, 5], [5, 5, 5, 5]],
    'num_iterations': [10, 100, 1000]
}

class GridSearch():
    def __init__(self, estimator, param_grid):
        self.estimator = deepcopy(estimator)
        self.param_grid = param_grid

    def fit(self, X, y):
        def iterate(param_grid, dim, param_values):
            if dim == len(param_grid.keys()):
                for key in param_values:
                    setattr(self.estimator, key, param_values[key])
                
                self.estimator.fit(X, y)
                score = self.estimator.score(X, y)
                self.param_scores_[str(param_values)] = score
            else:
                dim_key = list(param_grid.keys())[dim]
                dim_values = param_grid[dim_key]
                for value in dim_values:
                    param_values[dim_key] = value
                    iterate(param_grid, dim + 1, param_values)
        
        self.param_scores_ = {}
        
        iterate(self.param_grid, 0, {})
        self.best_params_ = eval(max(self.param_scores_, \
            key = self.param_scores_.get))

        for key in self.best_params_:
            setattr(self.estimator, key, self.best_params_[key])
        self.best_estimator_ = deepcopy(self.estimator)
        
# %%
from mlp import MLPClassifier
import pandas as pd

gs = GridSearch(MLPClassifier(), param_grid)

train = pd.read_csv("data/projekt1_test/Classification/data.circles.train.100.csv")
test = pd.read_csv("data/projekt1_test/Classification/data.circles.test.100.csv")
X_train, y_train = train.iloc[:, :-1], train.cls
X_test, y_test = test.iloc[:, :-1], test.cls

gs.fit(X_train, y_train)

# %%
gs.best_params_

# %%
gs.param_scores_

# %%
