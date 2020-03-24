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


clf = MLPClassifier(num_iterations=7200, eta=0.3805, \
    batch_portion=0.3, bias=True, hidden_layers=[196, 196],\
    activation_function=activation_functions.sigmoid, \
    error_function=error_functions.cross_entropy, \
    moment=0.7605, \
    random_seed = 12369666)
clf = clf.fit(X_train, y_train)

# %%
import joblib
joblib.dump(clf, 'clf_digits.joblib')

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
import pandas as pd
X = pd.read_csv("data/digits/test.csv.zip")
predicted_y = clf.predict(X)

result = pd.DataFrame({
        'Label': predicted_y
    })\
    .reset_index(drop = False)\
    .rename(columns = { 'index': 'ImageId' })
result['ImageId'] = result['ImageId'] + 1
result.to_csv('kaggle_result.csv', index = False)

# %%
