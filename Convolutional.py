from turtle import forward
import numpy as np
from scipy import signal
from Layer import Layer

class Convolutional(Layer):
    def __init__(self, input_shape, kernel_size, depth):
        input_depth, input_height, input_width = input_shape
        self.depth = depth
        self.input_shape = input_shape
        self.input_depth = input_depth
        self.output_shape = (depth, input_height - kernel_size + 1, input_width - kernel_size + 1)
        self.kernel_shape = (depth, input_depth, kernel_size, kernel_size)
        self.kernels = np.random.randn(*self.kernel_shape)
        self.biases = np.random.randn(*self.output_shape)
        super().__init__()

    def forward(self, input):
        self.input = input
        self.output = np.copy(self.biases)
        for i in range(self.depth):
            for j in range(self.input_depth):
                self.output[i] += signal.correlate2d(self.input[j], self.kernels[i,j], "valid") 



    def backward(self, output_grad, alpha):
        kernel_grad = np.zeros(self.kernel_shape)
        input_grad = np.zeros(self.input_shape)

        for i in range(self.depth):
            for j in range(self.input_depth):
                kernel_grad[i, j] = signal.correlate2d(self.input[j], output_grad[i], "valid")
                input_grad[j] += signal.convolve2d(output_grad[i], self.kernels[i,j], "full")


        self.kernels -= alpha * kernel_grad
        self.biases -= alpha * output_grad
        return input_grad                        