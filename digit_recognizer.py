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
    moment = 0.7, \
    random_seed = 12369666)
clf = clf.fit(X_train, y_train, \
    serialize_path = 'clf_digits.joblib')

# %%
print("Train accuracy:", clf.score(X_train, y_train))

# %%
print("Test accuracy:", clf.score(X_test, y_test))

# %%
print("Confusion matrix:")
print(clf.confusion_matrix(X_test, y_test))

# %%
import joblib
clf = joblib.load('clf_digits.joblib')

# %%
X = pd.read_csv("data/digits/test.csv.zip")
predicted_y = clf.predict(X)

result = pd.DataFrame({
        'Label': predicted_y
    })\
    .reset_index(drop = False)\
    .rename(columns = { 'index': 'ImageId' })

result.to_csv('kaggle_result.csv', index = False)

# %%
