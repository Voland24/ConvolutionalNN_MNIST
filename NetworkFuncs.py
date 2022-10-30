#NN should be able to predict and train

def predict(network, input):
    output = input
    for layer in network:
        output = layer.forward(output)
    return output

def train(network, loss_func, loss_func_prim, x_train, y_train, epochs = 1000, alpha = 0.01, verbose = True):
    for e in range(epochs):
        error = 0
        for x,y in zip(x_train, y_train):

            output = predict(network, x)

            error += loss_func(y, output)

            grad = loss_func_prim(y, output)
            for layer in reversed(network):
                grad = layer.backward(grad, alpha)

        error /= len(x_train)
        if verbose:
            print(f"{e+1}/{epochs}, error = {error}")        