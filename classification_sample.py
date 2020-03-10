# %%
import pandas as pd

train = pd.read_csv("data/classification/data.simple.train.100.csv")
test = pd.read_csv("data/classification/data.simple.test.100.csv")
X_train, y_train = train.iloc[:, :-1], train.cls
X_test, y_test = test.iloc[:, :-1], test.cls

# %%
import matplotlib.pyplot as plt
plt.scatter(X_train.x, X_train.y, c = y_train)

# %%
import matplotlib.pyplot as plt
plt.scatter(X_test.x, X_test.y, c = y_test)

# %%
from mlp.mlp_classifier import MLPClassifier

clf = MLPClassifier()
clf.fit(X_train, y_train).score(X_test, y_test)

# %%
train_errors = clf.error_by_iteration(X_train.values, y_train.values)
test_errors = clf.error_by_iteration(X_test.values, y_test.values)

# %%
import matplotlib.pyplot as plt
plt.yscale('log')
plt.plot(train_errors)
plt.plot(test_errors)
plt.show()

# %%
from mlp.mlp_classifier import MLPClassifier
import mlp.activation_functions as activation_functions

clf = MLPClassifier(activation_function = activation_functions.sigmoid)
clf.fit(X_train, y_train).score(X_test, y_test)

# %%
from mlp.mlp_classifier import MLPClassifier
import mlp.activation_functions as activation_functions
import mlp.error_functions as error_functions

clf = MLPClassifier(error_function = error_functions.mean_squared)
clf.fit(X_train, y_train).score(X_test, y_test)