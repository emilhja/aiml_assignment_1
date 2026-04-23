#!/usr/bin/env python3
import numpy as np


class ReLU:
    def __call__(self, x):
        return np.maximum(0, x)

class Sigmoid:
    def __call__(self, x):
        return 1 / (1 + np.exp(-x))

class LeakyReLU:
    def __init__(self, alpha=0.01):
        self.alpha = alpha

    def __call__(self, x):
        return np.where(x > 0, x, self.alpha * x)

class Layer:
    def __init__(self, weights, bias, activation):
        self.weights = np.array(weights, dtype=float)
        self.bias = np.array(bias, dtype=float)
        self.activation = activation

    def forward(self, inputs):
        inputs = np.array(inputs, dtype=float)
        z = self.weights @ inputs + self.bias # The @ operator is a shorthand for np.dot() when used with 2D arrays, and it performs matrix multiplication. In this case, it multiplies the weights matrix with the input vector, which is exactly what we want to do in a fully connected layer.
        return self.activation(z)

# A is vector
A = [2.0, -3.0, 4.5]
# W is matrix (3 neurons, each with 3 weights), shape: (3, 3)
W = [
    [0.5, 1.2, -2.0],
    [1.0, -0.5, 0.8],
    [-1.5, 2.0, 0.3],
]
B = [1.2, -0.7, 0.5]

layer = Layer(W, B, LeakyReLU())
# y = (3, ) * (3,3) = (3, ) -> the output is a vector with 3 elements, one for each neuron in the layer
y = layer.forward(A)

print(y)
