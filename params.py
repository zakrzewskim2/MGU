from enum import Enum

class ProblemType(Enum):
    CLASSIFICATION = 1
    REGRESSION = 2

class ActivationFunction(Enum):
    SIGMOID = 1
    TANH = 2
    LINEAR = 3
    SOFTMAX = 4

hidden_layers = [5, 5, 3]
activation_function = ActivationFunction.SIGMOID
bias = True
batch_size = 0.3
num_iterations = 100000
eta = 0.1
momentum = 0.1
problem_type = ProblemType.CLASSIFICATION