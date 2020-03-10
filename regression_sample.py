# %%
import pandas as pd

train = pd.read_csv("data/regression/data.cube.train.100.csv")
test = pd.read_csv("data/regression/data.cube.test.100.csv")
X_train, y_train = train.iloc[:, :-1], train.y
X_test, y_test = test.iloc[:, :-1], test.y

# %%
import matplotlib.pyplot as plt
plt.scatter(X_train, y_train)

# %%
import matplotlib.pyplot as plt
plt.scatter(X_test, y_test)

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
