#used to reshape the 3D output of the CNN into a column vector for Dense Layer


import numpy as np
from Layer import Layer

class Reshape(Layer):
    def __init__(self, input_shape, output_shape):
        self.input_shape = input_shape
        self.output_shape = output_shape
        super().__init__()


    def forward(self, input):
        return np.reshape(input, self.output_shape)

    def backward(self, output_grad, alpha):
        return np.reshape(output_grad, self.input_shape)        