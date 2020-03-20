# %%
import pandas as pd

from grid_search import GridSearch
from mlp import MLPClassifier

param_grid = {
    'hidden_layers': [[], [5], [5, 5], [5, 5, 5], [5, 5, 5, 5]],
    'num_iterations': [10, 100, 1000]
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
