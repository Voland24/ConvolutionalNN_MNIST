#fully connected layer
#using the matrix mul form to simplify

#y = W * x + b

from Layer import Layer
import numpy as np

class Dense(Layer):
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(output_size, input_size)
        self.bias = np.random.randn(output_size, 1)
        super().__init__()

    def forward(self, input):
        self.input = input
        return np.dot(self.weights, self.input) + self.bias


    def backward(self, output_grad, alpha):
        #TODO grad of weights and biases and grad of input for backward pass
        weights_grad = np.dot(output_grad, self.input.T)
        self.weights -= alpha * weights_grad
        #bias_grad is the same as output grad
        self.bias -= alpha * output_grad
        return np.dot(self.weights.T, output_grad)        