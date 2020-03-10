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



#%%
import seaborn as sns

df = sns.load_dataset('diamonds')
df
train = df.sample(frac=0.7, random_state=200)
test = df.drop(train.index)
X_train_rgs, y_train_rgs = train.loc[:200, ['carat', 'depth']], train.price
X_test_rgs, y_test_rgs = test.loc[:200, ['carat', 'depth']], test.price

# %%
from mlp.mlp_regressor import MLPRegressor

est = MLPRegressor()
est.fit(X_train_rgs, y_train_rgs).score(X_test_rgs, y_test_rgs)

# %%
import seaborn as sns

df = sns.load_dataset('tips')
df
train = df.sample(frac=0.7, random_state=200)
test = df.drop(train.index)
X_train, y_train = train.loc[:200, ['total_bill']], train.loc[:, 'size']
X_test, y_test = test.loc[:200, ['total_bill']], test.loc[:, 'size']

# %%
from mlp.mlp_classifier import MLPClassifier
import mlp.activation_functions as activation_functions

clf = MLPClassifier(activation_function = activation_functions.tanh)
clf.fit(X_train, y_train).score(X_test, y_test)

# %%
