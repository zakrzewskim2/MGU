# %%
import pandas as pd

train = pd.read_csv("data/classification/data.simple.train.100.csv")
test = pd.read_csv("data/classification/data.simple.test.100.csv")
X_train, y_train = train.iloc[:, :-1], train.cls
X_test, y_test = test.iloc[:, :-1], test.cls

# %%
from mlp.mlp_classifier import MLPClassifier

clf = MLPClassifier()
clf.fit(X_train, y_train).score(X_test, y_test)

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

# %%
