#Used as a Base layer for all other layers types of layers

#It has a method for forward pass

#Also, a method for backpass

#It takes in the derivative from the layer in front of it
#calculates the derivative with respect to it's own weights
#and the derivative with respect to it's input and passes it back


class Layer:
    def __init__(self):
        self.input = None
        self.output = None


    def forward(self, input):
        #TODO return the output
        pass    

    def backward(self, output_grad, alpha):
        #TODO update params and return input grad
        #the alpha can be replaced by an optimizer function
        pass