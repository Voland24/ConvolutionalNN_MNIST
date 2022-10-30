#layer for activation functions

#moved to not be in Dense layer, to not complicate calculations

from Layer import Layer
import numpy as np

class Activation(Layer):
    def __init__(self, activation, activation_prim):
        self.activation = activation
        self.activation_prim = activation_prim
        super().__init__()

    def forward(self, input):
        self.input = input
        return self.activation(self.input)


    def backward(self, output_grad, alpha):
        return np.multiply(output_grad, self.activation_prim(self.input)) 
        


