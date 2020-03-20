from copy import deepcopy

class GridSearch():
    def __init__(self, estimator, param_grid):
        self.estimator = deepcopy(estimator)
        self.param_grid = param_grid

    def fit(self, X, y):
        def iterate(param_grid, dim, param_values):
            if dim == len(param_grid.keys()):
                for key in param_values:
                    setattr(self.estimator, key, param_values[key])
                
                print(param_values)
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
