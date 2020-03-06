from enum import Enum
import activation_functions
import error_functions


class ProblemType(Enum):
    CLASSIFICATION = 1
    REGRESSION = 2


activation_function = activation_functions.softmax
error_function = error_functions.cross_entropy

hidden_layers = [5, 5, 3]
bias = True
batch_size = 0.3
num_iterations = 100000
eta = 0.1
momentum = 0.1
problem_type = ProblemType.CLASSIFICATION
