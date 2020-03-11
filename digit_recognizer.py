# %%
import pandas as pd

train = pd.read_csv("data/digits/train.csv.zip")
test = pd.read_csv("data/digits/test.csv.zip")
X_train, y_train = train.iloc[:500, 1:], train.iloc[:500, 0]
X_test, y_test = test.iloc[:, 1:], test.iloc[:, 0]

# %%
from mlp.mlp_classifier import MLPClassifier

clf = MLPClassifier()
clf.fit(X_train, y_train).score(X_test, y_test)


# %%
