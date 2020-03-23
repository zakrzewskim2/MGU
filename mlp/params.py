import mlp.activation_functions as activation_functions
import mlp.error_functions as error_functions

class Config():
    def __init__(self):
        self.activation_function = activation_functions.softmax
        self.out_activation_function = activation_functions.softmax
        self.error_function = error_functions.cross_entropy

    hidden_layers = [5, 5, 3]
    bias = True
    batch_portion = 0.4
    num_iterations = 100000
    eta = 0.1
    moment = 0.9