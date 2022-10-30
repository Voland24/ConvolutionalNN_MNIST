from distutils.debug import DEBUG
import numpy as np
from keras.datasets import mnist
from keras.utils import np_utils

from Dense import Dense
from Convolutional import Convolutional
from Reshape import Reshape
from ActivationFuncs import Sigmoid
from LossFuncs import binary_cross_entropy, binary_cross_entropy_prim
from NetworkFuncs import train, predict

def data_prep(x,y,limit):
    zero_index = np.where(y == 0)[0][:limit]
    one_index = np.where(y == 1)[0][:limit]
    all_indices = np.hstack((zero_index, one_index))
    all_indices = np.random.permutation(all_indices)
    x,y = x[all_indices], y[all_indices]
    x = x.reshape(len(x), 1 , 28 , 28)
    x = x.astype("float32") / 255
    y = np_utils.to_categorical(y)
    y = y.reshape(len(y), 2, 1)
    return x, y



(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, y_train = data_prep(x_train, y_train, 100)
x_test, y_test = data_prep(x_test, y_test, 100)

network = [
    Convolutional((1,28,28),3,5),
    Sigmoid(),
    Reshape((5,26,26),(5*26*26,1)),
    Dense(5*26*26,100),
    Sigmoid(),
    Dense(100,2),
    Sigmoid()
]

train(
    network,
    binary_cross_entropy,
    binary_cross_entropy_prim,
    x_train,
    y_train,
    epochs=20,
    alpha=0.1
)

for x,y in zip(x_test, y_test):
    output = predict(network, x)
    print(f"pred: {np.argmax(output)}, true: {np.argmax(y)}")
    