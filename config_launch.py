# %%
import yaml, sys, os

config = None
with open('config.yml', 'r') as f:
    try:
        config = yaml.safe_load(f)
    except yaml.YAMLError as exc:
        print(exc)
        sys.exit()

if 'train_set_path' not in config \
        or 'test_set_path' not in config \
        or 'problem_type' not in config \
        or 'hidden_layers' not in config \
        or 'activation_function' not in config \
        or 'error_function' not in config \
        or 'iterations' not in config \
        or 'learning_rate' not in config \
        or 'moment' not in config \
        or 'random_seed' not in config \
        or 'bias' not in config \
        or 'batch_portion' not in config:
    print('Missing key in configuration')
    sys.exit()

if type(config['train_set_path']) != str or not os.path.exists(config['train_set_path']):
    print('Invalid train_set_path')
    sys.exit()
if type(config['test_set_path']) != str or not os.path.exists(config['test_set_path']):
    print('Invalid test_set_path')
    sys.exit()

if config['problem_type'] not in ['classification', 'regression']:
    print('Invalid problem_type')
    sys.exit()
if type(config['hidden_layers']) != list:
    print('Invalid hidden_layers')
    sys.exit()
else:
    for element in config['hidden_layers']:
        if type(element) != int:
            print('Invalid hidden_layers')
            sys.exit()
if config['activation_function'] not in ['sigmoid', 'linear', 'tanh', 'softmax']:
    print('Invalid activation_function')
    sys.exit()
if config['error_function'] not in ['mean_squared', 'mean', 'max', 'cross_entropy']:
    print('Invalid error_function')
    sys.exit()
if type(config['iterations']) != int or config['iterations'] <= 0:
    print('Invalid iterations')
    sys.exit()
if type(config['learning_rate']) not in [int, float] or config['learning_rate'] <= 0:
    print('Invalid learning_rate')
    sys.exit()
if type(config['moment']) not in [int, float] or config['moment'] < 0:
    print('Invalid moment')
    sys.exit()
if config['random_seed'] is not None and type(config['random_seed']) != int:
    print('Invalid random_seed')
    sys.exit()
if type(config['bias']) != bool:
    print('Invalid bias')
    sys.exit()
if type(config['batch_portion']) not in [int, float] or config['batch_portion'] < 0 or config['batch_portion'] > 1:
    print('Invalid batch_portion')
    sys.exit()

# %%
import pandas as pd

try:
    train = pd.read_csv(config['train_set_path'])
except:
    print('Invalid train dataset format')
    sys.exit()
    
try:
    test = pd.read_csv(config['test_set_path'])
except:
    print('Invalid train dataset format')
    sys.exit()

X_train, y_train = train.iloc[:, :-1], train.iloc[:, -1]
X_test, y_test = test.iloc[:, :-1], test.iloc[:, -1]

# %%
from mlp import MLPClassifier, MLPRegressor, Visualizer, \
    activation_functions, error_functions

vis = Visualizer()

activation_mapping = {
    'sigmoid': activation_functions.sigmoid, \
    'linear': activation_functions.linear, \
    'softmax': activation_functions.softmax, \
    'tanh': activation_functions.tanh
}
activation_function = activation_mapping.get(config['activation_function'])

error_mapping = {
    'mean_squared': error_functions.mean_squared, \
    'mean': error_functions.mean, \
    'max': error_functions.max_error, \
    'cross_entropy': error_functions.cross_entropy
}
error_function = error_mapping.get(config['error_function'])

if config['problem_type'] == 'classification':
    clf = MLPClassifier(num_iterations = config['iterations'], \
        bias = config['bias'], \
        hidden_layers = config['hidden_layers'], \
        eta = config['learning_rate'], \
        moment = config['moment'], \
        batch_portion = config['batch_portion'], \
        random_seed = config['random_seed'], \
        activation_function = activation_function, \
        error_function = error_function)
    clf = clf.fit(X_train, y_train, \
        serialize_path='training_data.joblib')

    print('Classification accuracy:', clf.score(X_test, y_test))
    print('Class confusion matrix:')
    print(clf.confusion_matrix(X_test, y_test))    

    print('Plotting training dataset...')
    vis.plot_classification_dataset(X_train.iloc[:, 0], X_train.iloc[:, 1], y_train, show = True)
    print('Plotting test dataset...')
    vis.plot_classification_dataset(X_test.iloc[:, 0], X_test.iloc[:, 1], y_test, show = True)
    print('Plotting classifier decision space for train data...')
    vis.plot_classification_result(clf, X_train.iloc[:, 0], X_train.iloc[:, 1], y_train, show = True)
    print('Plotting classifier decision space for test data...')
    vis.plot_classification_result(clf, X_test.iloc[:, 0], X_test.iloc[:, 1], y_test, show = True)
    print('Computing and plotting errors on train and test datasets for each iteration... (might take a while)')
    vis.plot_train_test_error(clf, X_train, y_train, X_test, y_test, show = True)
    print('Plotting edge weights during training...')
    vis.plot_training_history('training_data.joblib')
    print('Finished')

if config['problem_type'] == 'regression':
    estimator = MLPRegressor(num_iterations = config['iterations'], \
        bias = config['bias'], \
        hidden_layers = config['hidden_layers'], \
        eta = config['learning_rate'], \
        moment = config['moment'], \
        batch_portion = config['batch_portion'], \
        random_seed = config['random_seed'], \
        activation_function = activation_function, \
        error_function = error_function)
    estimator = estimator.fit(X_train, y_train, \
        serialize_path='training_data.joblib')

    print('Regression R^2 score:', estimator.score(X_test, y_test))

    print('Plotting training dataset...')
    vis.plot_regression_dataset(X_train.iloc[:, 0], y_train, show = True)
    print('Plotting test dataset...')
    vis.plot_regression_dataset(X_test.iloc[:, 0], y_test, show = True)
    print('Plotting classifier decision space for train data...')
    vis.plot_regression_result(estimator, X_train.iloc[:, 0], y_train, show = True)
    print('Plotting classifier decision space for test data...')
    vis.plot_regression_result(estimator, X_test.iloc[:, 0], y_test, show = True)
    print('Computing and plotting errors on train and test datasets for each iteration... (might take a while)')
    vis.plot_train_test_error(estimator, X_train, y_train, X_test, y_test, show = True)
    print('Plotting edge weights during training...')
    vis.plot_training_history('training_data.joblib')
    print('Finished')
