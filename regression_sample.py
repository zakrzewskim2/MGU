# %%
import pandas as pd
import numpy as np
train = pd.read_csv("data/regression/data.activation.train.100.csv")
test = pd.read_csv("data/regression/data.activation.test.100.csv")

X_train, y_train = train.iloc[:, :-1], train.y
X_test, y_test = test.iloc[:, :-1], test.y

# %%
from mlp import MLPRegressor

estimator = MLPRegressor(random_seed = 12369666, \
    num_iterations = 10000)
print('R^2 score:', estimator.fit(X_train, y_train).score(X_test, y_test))

# %%
from mlp import Visualizer
vis = Visualizer()
vis.plot_train_test_error(estimator, X_train, y_train, \
    X_test, y_test, log_scale=True, show = True)

# %%
from mlp import Visualizer
vis = Visualizer()
vis.plot_regression_result(estimator, X_test.x, y_test, show = True)

# %%
from mlp import Visualizer
vis = Visualizer()
vis.plot_regression_dataset(X_train, y_train, show = True)

# %%
from mlp import Visualizer
vis = Visualizer()
vis.plot_regression_dataset(X_test, y_test, show = True)

# %%
