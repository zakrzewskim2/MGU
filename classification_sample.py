# %%
import pandas as pd

train = pd.read_csv("data/projekt1_test/Classification/data.circles.train.100.csv")
test = pd.read_csv("data/projekt1_test/Classification/data.circles.test.100.csv")
X_train, y_train = train.iloc[:, :-1], train.cls
X_test, y_test = test.iloc[:, :-1], test.cls

# %%
from mlp.mlp_classifier import MLPClassifier

clf = MLPClassifier()
clf.fit(X_train, y_train, serialize_path='training_data.joblib').score(X_test, y_test)

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

# %%
import matplotlib.pyplot as plt
import numpy as np

def decision_space_plot(clf, x, y, real_class):
    x_min, x_max = np.min(x) - 1, np.max(x) + 1
    y_min, y_max = np.min(y) - 1, np.max(y) + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))

    z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    z = z.reshape(xx.shape)

    plt.contourf(xx, yy, z, alpha=0.4)
    plt.scatter(x, y, c=real_class, s=20, edgecolor='k')

# decision_space_plot(clf, X_train.x, X_train.y, y_train)
decision_space_plot(clf, X_test.x, X_test.y, y_test)

# %%
import matplotlib.pyplot as plt
plt.scatter(X_train.x, X_train.y, c = y_train)

# %%
import matplotlib.pyplot as plt
plt.scatter(X_test.x, X_test.y, c = y_test)