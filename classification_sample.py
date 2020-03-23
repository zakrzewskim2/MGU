# %%
import pandas as pd

train = pd.read_csv("data/projekt1_test/Classification/data.circles.train.100.csv")
test = pd.read_csv("data/projekt1_test/Classification/data.circles.test.100.csv")
X_train, y_train = train.iloc[:, :-1], train.cls
X_test, y_test = test.iloc[:, :-1], test.cls

# %%
from mlp import MLPClassifier

clf = MLPClassifier(num_iterations = 10000, \
    random_seed = 12369666)
print('Accuracy:', clf.fit(X_train, y_train, \
        serialize_path='training_data.joblib')\
    .score(X_test, y_test))

# %%
print('Class confusion matrix:')
print(clf.confusion_matrix(X_test, y_test))

# %%
from mlp import Visualizer
vis = Visualizer()
vis.plot_train_test_error(clf, X_train, y_train, \
    X_test, y_test, log_scale=True, show = True)

# %%
from mlp import MLPClassifier, activation_functions

clf = MLPClassifier(activation_function = activation_functions.sigmoid, \
    random_seed = 12369666)
clf.fit(X_train, y_train).score(X_test, y_test)

# %%
from mlp import MLPClassifier, activation_functions, \
    error_functions

clf = MLPClassifier(error_function = error_functions.mean_squared, \
    random_seed = 12369666)
clf.fit(X_train, y_train).score(X_test, y_test)

# %%
from mlp import Visualizer
vis = Visualizer()
vis.plot_classification_result(clf, X_test.x, \
    X_test.y, y_test, show = True)

# %%
from mlp import Visualizer
vis = Visualizer()
vis.plot_classification_dataset(X_train.x, X_train.y, \
    y_train, show = True)

# %%
from mlp import Visualizer
vis = Visualizer()
vis.plot_classification_dataset(X_test.x, X_test.y, \
    y_test, show = True)

# %%
from mlp import Visualizer
vis = Visualizer()
vis.plot_training_history(filename = 'training_data.joblib')

# %%
