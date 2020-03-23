# %%
import pandas as pd
import numpy as np
import random as rand

train = pd.read_csv("data/digits/train.csv.zip")
test = pd.read_csv("data/digits/test.csv.zip")
indices = np.random.permutation(train.shape[0])
train_indices, test_indices = indices[:10000], indices[10000:]

X_train, y_train = train.iloc[train_indices, 1:], train.iloc[train_indices, 0]
X_test, y_test = train.iloc[test_indices, 1:], train.iloc[test_indices, 0]

X_train /= 255
X_test /= 255

# %%
from mlp import MLPClassifier, activation_functions, \
    error_functions

clf = MLPClassifier(activation_function = activation_functions.sigmoid, \
    error_function = error_functions.cross_entropy, \
    hidden_layers = [49, 28, 16],
    bias = True, \
    batch_portion = 0.5, \
    num_iterations = 10000, \
    eta = 0.15, \
    moment = 0.7)
clf = clf.fit(X_train, y_train)

# %%
import joblib
joblib.dump(clf, 'clf_digits.joblib')

# %%
print("train", clf.score(X_train, y_train))

# %%
print("test", clf.score(X_test, y_test))

# %%
clf.confusion_matrix(X_test, y_test)

# %%
