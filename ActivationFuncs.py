from Activation import Activation
import numpy as np

class Tanh(Activation):
    def __init__(self):
        tanh = lambda x : np.tanh(x)
        tanh_prim = lambda x : 1 - tanh(x) ** 2
        super().__init__(tanh, tanh_prim)


class Sigmoid(Activation):
    def __init__(self):
        sigmoid = lambda x : (1 / (1 + np.exp(-x)))
        sigmoid_prim = lambda x: sigmoid(x) * (1 - sigmoid(x))
        super().__init__(sigmoid, sigmoid_prim)



