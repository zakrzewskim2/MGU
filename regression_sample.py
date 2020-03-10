# %%
import pandas as pd

train = pd.read_csv("data/regression/data.cube.train.100.csv")
test = pd.read_csv("data/regression/data.cube.test.100.csv")
X_train, y_train = train.iloc[:, :-1], train.y
X_test, y_test = test.iloc[:, :-1], test.y

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# %%
from mlp.mlp_regressor import MLPRegressor

estimator = MLPRegressor()
estimator.fit(X_train, y_train).score(X_test, y_test)

# %%
train_errors = estimator.error_by_iteration(X_train.values, y_train.values)
test_errors = estimator.error_by_iteration(X_test.values, y_test.values)

# %%
import matplotlib.pyplot as plt
plt.yscale('log')
plt.plot(train_errors)
plt.plot(test_errors)

# %%
import matplotlib.pyplot as plt
import numpy as np

def regression_line_plot(estimator, x, y):
    x_min, x_max = np.min(x) - 1, np.max(x) + 1
    xx = np.arange(x_min, x_max, 0.1)

    yy = estimator.predict(xx.reshape(-1, 1))
    yy = yy.reshape(xx.shape)

    plt.scatter(x, y, s = 5, edgecolor='k')
    plt.plot(xx, yy, 'r--')

regression_line_plot(estimator, X_test[:, 0], y_test)

# %%
import matplotlib.pyplot as plt
plt.scatter(X_train, y_train)

# %%
import matplotlib.pyplot as plt
plt.scatter(X_test, y_test)

# %%
