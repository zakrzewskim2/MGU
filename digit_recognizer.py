# %%
import pandas as pd
import numpy as np
import random as rand

train = pd.read_csv("data/digits/train.csv.zip")
test = pd.read_csv("data/digits/test.csv.zip")
indices = np.random.permutation(train.shape[0])
train_indices, test_indices = indices[:26000], indices[26000:]

X_train, y_train = train.iloc[train_indices, 1:], train.iloc[train_indices, 0]
X_test, y_test = train.iloc[test_indices, 1:], train.iloc[test_indices, 0]

X_train /= 255
X_test /= 255

# %%
from mlp.mlp_classifier import MLPClassifier

clf = MLPClassifier()
clf = clf.fit(X_train, y_train)

# %%
print("train", clf.score(X_train, y_train))

# %%
print("test", clf.score(X_test, y_test))

# %%
