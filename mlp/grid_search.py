from copy import deepcopy

class GridSearch():
    def __init__(self, estimator, param_grid):
        self.estimator = deepcopy(estimator)
        self.param_grid = param_grid

    def fit(self, X_train, y_train, X_test, y_test):
        def iterate(param_grid, dim, param_values):
            if dim == len(param_grid.keys()):
                for key in param_values:
                    setattr(self.estimator, key, param_values[key])
                
                self.estimator.fit(X_train, y_train)
                score = self.estimator.score(X_test, y_test)
                self.param_scores_.append({
                    'params': deepcopy(param_values), 
                    'score': score
                })
            else:
                dim_key = list(param_grid.keys())[dim]
                dim_values = param_grid[dim_key]
                for value in dim_values:
                    param_values[dim_key] = value
                    iterate(param_grid, dim + 1, param_values)
        
        self.param_scores_ = []

        iterate(self.param_grid, 0, {})
        self.best_params_ = max(self.param_scores_, \
            key = lambda p: p['score'])['params']

        for key in self.best_params_:
            setattr(self.estimator, key, self.best_params_[key])
        self.estimator.fit(X_train, y_train)
        self.best_estimator_ = deepcopy(self.estimator)
