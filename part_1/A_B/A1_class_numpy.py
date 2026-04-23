#!/usr/bin/env python3
# neuron.py

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

class Neuron:
    def __init__(self, weights, bias, activation):
        self.weights = np.array(weights, dtype=float) # it ensures weights/inputs are treated as decimal numbers, which is usually what you want in neural network code
        self.bias = bias
        self.activation = activation

    def forward(self, inputs):
        inputs = np.array(inputs, dtype=float) # it ensures weights/inputs are treated as decimal numbers, which is usually what you want in neural network code
        z = np.dot(inputs, self.weights) + self.bias

        return self.activation(z)

A = [2.0, -3.0, 4.5]  # indata
W = [0.5, 1.2, -2.0]  # Model weights for this neuron
B = 1.2

neuron = Neuron(W, B, LeakyReLU())
y = neuron.forward(A)

print(f"{y}")
